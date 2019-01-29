from collections import deque

import tensorflow as tf
import numpy as np
import os
import label_map_util

MIN_SCORE_THRESHOLD = .50

HERE = os.path.dirname(os.path.abspath(__file__))


SIM_SSD_INCEPTION_PATH = os.path.join(HERE, 'models', 'sim_frozen_inference_graph.pb')
CARLA_SSD_INCEPTION_PATH = os.path.join(HERE, 'models', 'carla_frozen_inference_graph.pb')
LABELS_FILE = os.path.join(HERE, 'class_labels.pbtxt' )

class ObjectDetectionClassifier(object):
    def __init__(self, is_sim):
        # set default value for no detection

        model_path = SIM_SSD_INCEPTION_PATH if is_sim else CARLA_SSD_INCEPTION_PATH
        if not os.path.exists(model_path) or not os.path.exists(LABELS_FILE):
            raise ValueError("Unable to find the model path or labels file.")

        self.label_map = label_map_util.load_labelmap(LABELS_FILE)
        number_of_items = ("%s" %self.label_map).count("item") # we count them in the file
        self.categories = label_map_util.convert_label_map_to_categories(
            self.label_map, max_num_classes=number_of_items, use_display_name=True
        )

        self.category_index = label_map_util.create_category_index(self.categories)

        self.image_np_deep = None
        self.detection_graph = self._get_tf_graph(model_path)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.detection_graph, config=config)

        # LOAD tensors for detection
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def _get_tf_graph(self, model_path):
        detection_graph = tf.Graph()

        with detection_graph.as_default(), tf.gfile.GFile(model_path, 'rb') as f:
            od_graph_def = tf.GraphDef()
            od_graph_def.ParseFromString(f.read())
            tf.import_graph_def(od_graph_def, name='')

        return detection_graph

    def _detect_object(self, image):
        image_expanded = np.expand_dims(image, axis=0)

        fetches = [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections]
        boxes, scores, classes, num = self.sess.run(fetches, feed_dict={self.image_tensor: image_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        return boxes, scores, classes

    def classify_image2(self, image):
        _, scores, classes = self._detect_object(image)

        accumulators = dict()

        for i in xrange(len(scores)):
            if scores[i] > MIN_SCORE_THRESHOLD:
                class_id = classes[i]
                detection_score = scores[i]

                if class_id not in accumulators:
                    accumulators[class_id] = deque()

                accumulators[class_id].append(detection_score)

        max_items = {k: sum(v) / len(v) for k, v in accumulators.items()}

        candidate = None
        max_value = 0.
        for k, v in max_items.items():
            if max_value < v:
                candidate = k

        return candidate

if __name__ == '__main__':
    import cv2

    # trigger on off to test
    is_sim = True

    classifier = ObjectDetectionClassifier(is_sim)

    image_path = os.path.join(HERE, 'test_sim.jpg' if is_sim else 'test_carla.jpg')
    bgr_img = cv2.imread(image_path)
    image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    class_id = classifier.classify_image2(image)
    print(image_path + " : %s" % class_id)