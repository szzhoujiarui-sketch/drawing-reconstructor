from typing import List, Tuple

import cv2
import numpy as np


class FeatureMatcher:
    def __init__(self, detector: str = "sift", ratio_thresh: float = 0.75):
        self.ratio_thresh = ratio_thresh
        if detector == "sift":
            self.detector = cv2.SIFT_create()
        elif detector == "orb":
            self.detector = cv2.ORB_create(nfeatures=4000)
        else:
            raise ValueError(f"Unsupported detector: {detector}")

        if detector == "sift":
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        kp, des = self.detector.detectAndCompute(gray, None)
        return kp, des

    def match(self, des1: np.ndarray, des2: np.ndarray) -> List[cv2.DMatch]:
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return []
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < self.ratio_thresh * n.distance:
                    good.append(m)
            elif len(m_n) == 1:
                good.append(m_n[0])
        return good

    @staticmethod
    def get_match_points(matches: List[cv2.DMatch], kp1, kp2) -> Tuple[np.ndarray, np.ndarray]:
        if not matches:
            return np.empty((0, 2)), np.empty((0, 2))
        valid = [
            m for m in matches
            if 0 <= m.queryIdx < len(kp1) and 0 <= m.trainIdx < len(kp2)
        ]
        if not valid:
            return np.empty((0, 2)), np.empty((0, 2))
        src = np.float32([kp1[m.queryIdx].pt for m in valid]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in valid]).reshape(-1, 1, 2)
        return src, dst
