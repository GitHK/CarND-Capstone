from styx_msgs.msg import TrafficLight
from ssd_inceptionv2.detector import ObjectDetectionClassifier
import cv2

FAST_TRANSLATION = {
    None : TrafficLight.UNKNOWN,
    1: TrafficLight.GREEN,
    2: TrafficLight.RED,
    3: TrafficLight.YELLOW,
    4: TrafficLight.UNKNOWN,
}

class TLClassifier(object):
    def __init__(self, is_sim):
        self.is_sim = is_sim
        self.classifier = ObjectDetectionClassifier(self.is_sim)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_number = self.classifier.classify_image2(img)

        return FAST_TRANSLATION[class_number]
