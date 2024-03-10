from pycocotools.cocoeval import COCOeval
import json
import torch
import numpy as np
import onnxruntime

def evaluate_coco_onnx(dataset,onnx_path, threshold=0.05):
    
        ort_session = onnxruntime.InferenceSession(onnx_path)
    
    # with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []
        for index in range(len(dataset)):
            data = dataset[index]
            xscale = data['xscale']
            yscale = data['yscale']
            # run network
            input_name = ort_session.get_inputs()[0].name
            output_names = [output.name for output in ort_session.get_outputs()]
            img_tensor = np.expand_dims(data['img'].permute(2, 0, 1).numpy().astype(np.float32), axis=0)  # Add batch dimension
            ort_inputs = {input_name: img_tensor}
            ort_outputs = ort_session.run(output_names, ort_inputs)
            scores, labels, boxes = ort_outputs
            # boxes /= scale
            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]
                
                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                print('no of boxes are',boxes.shape[0])
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset.image_ids[index],
                        'category_id' : dataset.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

        # write output
        json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # model.train()

        return
