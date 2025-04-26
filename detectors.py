from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import List, Tuple


class FeatureDetectorBase(ABC):
    @abstractmethod
    def match_features(self, image1: np.ndarray, image2: np.ndarray) -> List[
        Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """
        Match features between two images.

        Args:
            image1: First image (grayscale)
            image2: Second image (grayscale)

        Returns:
            List of tuples containing:
                ((x1, y1), (x2, y2), reliability)
            where:
                (x1, y1) - coordinates in first image
                (x2, y2) - coordinates in second image
                reliability - similarity score between 0 and 1
        """
        pass


class ORBDetector(FeatureDetectorBase):
    def __init__(self, nfeatures=500, scaleFactor=1.2, nlevels=8):
        self.detector = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match_features(self, image1: np.ndarray, image2: np.ndarray) -> List[
        Tuple[Tuple[float, float], Tuple[float, float], float]]:
        kp1, des1 = self.detector.detectAndCompute(image1, None)
        kp2, des2 = self.detector.detectAndCompute(image2, None)

        if des1 is None or des2 is None:
            return []

        matches = self.matcher.match(des1, des2)

        result = []
        for match in matches:
            x1, y1 = kp1[match.queryIdx].pt
            x2, y2 = kp2[match.trainIdx].pt
            reliability = 1 / (1 + match.distance)
            result.append(((float(x1), float(y1)), (float(x2), float(y2)), reliability))

        return result


class KAZEDetector(FeatureDetectorBase):
    def __init__(self, extended=False, upright=False):
        self.detector = cv2.KAZE_create(extended=extended, upright=upright)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def match_features(self, image1: np.ndarray, image2: np.ndarray) -> List[
        Tuple[Tuple[float, float], Tuple[float, float], float]]:
        kp1, des1 = self.detector.detectAndCompute(image1, None)
        kp2, des2 = self.detector.detectAndCompute(image2, None)

        if des1 is None or des2 is None:
            return []

        matches = self.matcher.match(des1, des2)

        result = []
        for match in matches:
            x1, y1 = kp1[match.queryIdx].pt
            x2, y2 = kp2[match.trainIdx].pt
            reliability = 1 / (1 + match.distance)
            result.append(((float(x1), float(y1)), (float(x2), float(y2)), reliability))

        return result


class AKAZEDetector(FeatureDetectorBase):
    def __init__(self, descriptor_type=cv2.AKAZE_DESCRIPTOR_KAZE, threshold=0.001):
        self.detector = cv2.AKAZE_create(descriptor_type=descriptor_type, threshold=threshold)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def match_features(self, image1: np.ndarray, image2: np.ndarray) -> List[
        Tuple[Tuple[float, float], Tuple[float, float], float]]:
        kp1, des1 = self.detector.detectAndCompute(image1, None)
        kp2, des2 = self.detector.detectAndCompute(image2, None)

        if des1 is None or des2 is None:
            return []

        matches = self.matcher.match(des1, des2)

        result = []
        for match in matches:
            x1, y1 = kp1[match.queryIdx].pt
            x2, y2 = kp2[match.trainIdx].pt
            reliability = 1 / (1 + match.distance)
            result.append(((float(x1), float(y1)), (float(x2), float(y2)), reliability))

        return result


class BRISKDetector(FeatureDetectorBase):
    def __init__(self, thresh=30, octaves=3):
        self.detector = cv2.BRISK_create(thresh=thresh, octaves=octaves)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def match_features(self, image1: np.ndarray, image2: np.ndarray) -> List[
        Tuple[Tuple[float, float], Tuple[float, float], float]]:
        kp1, des1 = self.detector.detectAndCompute(image1, None)
        kp2, des2 = self.detector.detectAndCompute(image2, None)

        if des1 is None or des2 is None:
            return []

        matches = self.matcher.match(des1, des2)

        result = []
        for match in matches:
            x1, y1 = kp1[match.queryIdx].pt
            x2, y2 = kp2[match.trainIdx].pt
            reliability = 1 / (1 + match.distance)
            result.append(((float(x1), float(y1)), (float(x2), float(y2)), reliability))

        return result


class SURFDetector(FeatureDetectorBase):
    def __init__(self, hessianThreshold=400, extended=False):
        self.detector = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold, extended=extended)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def match_features(self, image1: np.ndarray, image2: np.ndarray) -> List[
        Tuple[Tuple[float, float], Tuple[float, float], float]]:
        kp1, des1 = self.detector.detectAndCompute(image1, None)
        kp2, des2 = self.detector.detectAndCompute(image2, None)

        if des1 is None or des2 is None:
            return []

        matches = self.matcher.match(des1, des2)

        result = []
        for match in matches:
            x1, y1 = kp1[match.queryIdx].pt
            x2, y2 = kp2[match.trainIdx].pt
            reliability = 1 / (1 + match.distance)
            result.append(((float(x1), float(y1)), (float(x2), float(y2)), reliability))

        return result


class SIFTDetector(FeatureDetectorBase):
    def __init__(self, nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04):
        self.detector = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers,
                                        contrastThreshold=contrastThreshold)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def match_features(self, image1: np.ndarray, image2: np.ndarray) -> List[
        Tuple[Tuple[float, float], Tuple[float, float], float]]:
        kp1, des1 = self.detector.detectAndCompute(image1, None)
        kp2, des2 = self.detector.detectAndCompute(image2, None)

        if des1 is None or des2 is None:
            return []

        matches = self.matcher.match(des1, des2)

        result = []
        for match in matches:
            x1, y1 = kp1[match.queryIdx].pt
            x2, y2 = kp2[match.trainIdx].pt
            reliability = 1 / (1 + match.distance)
            result.append(((float(x1), float(y1)), (float(x2), float(y2)), reliability))

        return result