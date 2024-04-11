import numpy as np

def compute_oks(gts, preds, sigmas=None, area=1):
    """
    Compute Object Keypoint Similarity (OKS) for given ground truth and predicted keypoints.
    
    Parameters:
    - gts: NumPy array of ground truth keypoints with dimensions [17, 2].
    - preds: NumPy array of predicted keypoints with dimensions [17, 2].
    - sigmas: List or NumPy array of standard deviations for each keypoint.
    - area: The area of the object, used to normalize the OKS.
    
    Returns:
    - oks: The Object Keypoint Similarity between gts and preds.
    """
    
    if sigmas is None:
        # default keypoints from COCO dataset
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,
                           .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    
    vars = sigmas ** 2
    dx = preds[:, 0] - gts[:, 0]
    dy = preds[:, 1] - gts[:, 1]

    area = (np.max(gts[:, 0]) - np.min(gts[:, 0])) * (np.max(gts[:, 1]) - np.min(gts[:, 1]))
    
    e = (dx**2 + dy**2) / (vars* (area + np.spacing(1)))
    oks = np.sum(np.exp(-e)) / e.shape[0]
    
    return oks

