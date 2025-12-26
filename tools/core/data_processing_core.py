"""Data processing core utilities for gaze dataset preprocessing.

Based on GazeHub preprocessing pipeline.
Reference: https://phi-ai.buaa.edu.cn/Gazehub/
"""

import numpy as np
import cv2


class norm:
    """Image normalization for gaze estimation.
    
    Normalizes face images to a canonical view based on camera parameters
    and face/gaze annotations.
    """

    def __init__(
        self,
        center: np.ndarray,
        gazetarget: np.ndarray,
        headrotvec: np.ndarray,
        imsize: tuple = (224, 224),
        camparams: np.ndarray = None,
    ):
        """Initialize normalization.
        
        Args:
            center: 3D face center coordinates.
            gazetarget: 3D gaze target point.
            headrotvec: Head rotation vector (Rodrigues).
            imsize: Output image size (H, W).
            camparams: Camera intrinsic matrix (3x3).
        """
        self.center = np.array(center).reshape(3)
        self.gazetarget = np.array(gazetarget).reshape(3)
        self.headrotvec = np.array(headrotvec).reshape(3)
        self.imsize = imsize
        self.camera = camparams

        # Compute normalization parameters
        self._compute_transform()

    def _compute_transform(self):
        """Compute the normalization transformation matrix."""
        # Focal length for normalized camera
        focal_norm = 960
        distance_norm = 600
        
        # Current distance
        distance = np.linalg.norm(self.center)
        
        # Compute scale
        self.scale = distance_norm / distance
        
        # Compute rotation matrix to align face to camera
        z_axis = self.center / distance
        
        # Head rotation matrix
        head_rot, _ = cv2.Rodrigues(self.headrotvec)
        head_x = head_rot[:, 0]
        
        # Compute y-axis perpendicular to z and head_x
        y_axis = np.cross(z_axis, head_x)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Rotation matrix
        self.R = np.array([x_axis, y_axis, z_axis])
        
        # Normalized camera matrix
        self.camera_norm = np.array([
            [focal_norm, 0, self.imsize[0] / 2],
            [0, focal_norm, self.imsize[1] / 2],
            [0, 0, 1]
        ])
        
        # Scaling matrix
        self.S = np.diag([1.0, 1.0, self.scale])
        
        # Full transformation
        self.M = np.dot(self.camera_norm, np.dot(self.S, np.dot(self.R, np.linalg.inv(self.camera))))
        
        # Store params for later
        self.rvec = cv2.Rodrigues(self.R)[0].flatten()
        self.svec = np.array([self.scale, self.scale, self.scale])

    def GetImage(self, img: np.ndarray) -> np.ndarray:
        """Apply normalization to get normalized face image.
        
        Args:
            img: Input image.
            
        Returns:
            Normalized face image.
        """
        h, w = self.imsize
        normalized = cv2.warpPerspective(img, self.M, (w, h))
        return normalized

    def GetNewPos(self, point: np.ndarray) -> np.ndarray:
        """Get position of a 2D point in normalized image.
        
        Args:
            point: 2D point in original image.
            
        Returns:
            2D point in normalized image.
        """
        point = np.array(point).reshape(2)
        point_homo = np.array([point[0], point[1], 1.0])
        new_point = np.dot(self.M, point_homo)
        return new_point[:2] / new_point[2]

    def CropEye(self, left_corner: np.ndarray, right_corner: np.ndarray) -> np.ndarray:
        """Crop eye region from normalized face image.
        
        Args:
            left_corner: Left corner of eye.
            right_corner: Right corner of eye.
            
        Returns:
            Cropped eye image (60x36).
        """
        # Get eye center and size
        center = (left_corner + right_corner) / 2
        width = np.linalg.norm(right_corner - left_corner) * 1.5
        height = width * 0.6
        
        # Crop region
        x1 = int(max(0, center[0] - width / 2))
        x2 = int(min(self.imsize[0], center[0] + width / 2))
        y1 = int(max(0, center[1] - height / 2))
        y2 = int(min(self.imsize[1], center[1] + height / 2))
        
        # Get normalized image (stored during GetImage)
        # Note: This assumes GetImage was called first
        return np.zeros((36, 60, 3), dtype=np.uint8)  # Placeholder

    def GetGaze(self, scale: bool = True) -> np.ndarray:
        """Get normalized gaze direction.
        
        Args:
            scale: Whether to return as angles.
            
        Returns:
            Gaze direction (pitch, yaw) or 3D vector.
        """
        gaze_vec = self.gazetarget - self.center
        gaze_vec = gaze_vec / np.linalg.norm(gaze_vec)
        
        # Rotate to normalized coordinate
        gaze_norm = np.dot(self.R, gaze_vec)
        
        if scale:
            # Convert to pitch/yaw angles
            pitch = np.arcsin(-gaze_norm[1])
            yaw = np.arctan2(-gaze_norm[0], -gaze_norm[2])
            return np.array([pitch, yaw])
        else:
            return gaze_norm

    def GetHeadRot(self, vector: bool = True) -> np.ndarray:
        """Get normalized head rotation.
        
        Args:
            vector: Whether to return as rotation vector.
            
        Returns:
            Head rotation.
        """
        # Original head rotation matrix
        head_rot, _ = cv2.Rodrigues(self.headrotvec)
        
        # Rotate to normalized coordinate
        head_norm = np.dot(self.R, head_rot)
        
        if vector:
            # Convert back to rotation vector then to pitch/yaw
            rvec, _ = cv2.Rodrigues(head_norm)
            pitch = np.arcsin(-head_norm[1, 2])
            yaw = np.arctan2(head_norm[0, 2], head_norm[2, 2])
            return np.array([pitch, yaw])
        else:
            return head_norm

    def GetCoordinate(self, point: np.ndarray) -> np.ndarray:
        """Get 3D point in normalized coordinate."""
        point = np.array(point).reshape(3)
        return np.dot(self.R, point) * self.scale

    def GetParams(self) -> tuple:
        """Get normalization parameters (rvec, svec)."""
        return self.rvec, self.svec


def EqualizeHist(img: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to eye image.
    
    Args:
        img: Input BGR image.
        
    Returns:
        Histogram equalized image.
    """
    if len(img.shape) == 3:
        # Convert to YCrCb and equalize Y channel
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(img)


def GazeTo2d(gaze: np.ndarray) -> np.ndarray:
    """Convert 3D gaze vector to 2D angles (pitch, yaw).
    
    Args:
        gaze: 3D gaze vector or (pitch, yaw).
        
    Returns:
        2D gaze angles (pitch, yaw).
    """
    if len(gaze) == 2:
        return gaze
    
    # 3D vector to pitch/yaw
    pitch = np.arcsin(-gaze[1])
    yaw = np.arctan2(-gaze[0], -gaze[2])
    return np.array([pitch, yaw])


def HeadTo2d(head: np.ndarray) -> np.ndarray:
    """Convert head rotation to 2D angles."""
    if len(head) == 2:
        return head
    return head[:2]  # Already pitch/yaw


def GazeFlip(gaze: np.ndarray) -> np.ndarray:
    """Flip gaze direction horizontally."""
    gaze = np.array(gaze)
    gaze[1] = -gaze[1]  # Flip yaw
    return gaze


def HeadFlip(head: np.ndarray) -> np.ndarray:
    """Flip head direction horizontally."""
    head = np.array(head)
    head[1] = -head[1]  # Flip yaw
    return head
