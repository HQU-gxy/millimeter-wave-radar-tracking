from dataclasses import dataclass
from typing import Generator, Tuple, TypedDict, Optional
from pathlib import Path
from json import JSONEncoder, JSONDecoder
from os import PathLike

import cv2
import cv2 as cv
import numpy as np
import json
from loguru import logger
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

MatLike = NDArray
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
# https://github.com/google/jax/issues/10403


@dataclass
class LinearMotionNoInputModel:
    F: NDArray
    Q: NDArray


@dataclass
class LinearMeasurementModel:
    H: NDArray
    R: NDArray


Measurement = NDArray[np.float_]


@dataclass
class GaussianState:
    x: NDArray
    P: NDArray


def _predict(
    state: GaussianState,
    motion_model: LinearMotionNoInputModel,
) -> GaussianState:
    x = state.x
    P = state.P
    F = motion_model.F
    Q = motion_model.Q
    assert x.shape[0] == F.shape[
        0], "state and transition model are not compatible"
    assert F.shape[0] == F.shape[1], "transition model is not square"
    assert F.shape[0] == Q.shape[
        0], "transition model and noise model are not compatible"
    x_priori = F @ x
    P_priori = F @ P @ F.T + Q
    return GaussianState(x=x_priori, P=P_priori)


@dataclass
class PosterioriResult:
    # updated state
    state: GaussianState
    innovation: NDArray
    r"""
    y. Innovation refers to the difference between the observed measurement and the predicted measurement. Also known as the residual.

    .. math::
    
            y = z - H x_{\text{priori}}
    """
    innovation_covariance: NDArray
    r"""
    S. Innovation covariance refers to the covariance of the innovation (or residual) vector. 

    .. math::
    
            S = H  P H^T + R
    """
    posteriori_measurement: NDArray
    r"""
    z_posteriori. The updated measurement prediction.
    
    .. math::

        z_{\text{posteriori}} = H x_{\text{posteriori}}
    """
    mahalanobis_distance: NDArray
    r"""
    The Mahalanobis distance is a measure of the distance between a point P and a distribution D, introduced by P. Mahalanobis in 1936.

    .. math::
    
            \sqrt{y^T S^{-1} y}
    """
    squared_mahalanobis_distance: NDArray
    """
    If you are using the distance for statistical tests, such as identifying
    outliers, the squared Mahalanobis distance is often used because it corresponds
    to the chi-squared distribution when the underlying distribution is multivariate
    normal.
    """


def predict_measurement(
    state: GaussianState,
    measure_model: LinearMeasurementModel,
) -> Measurement:
    x = state.x
    H = measure_model.H
    return H @ x  # type: ignore


def update(
    measurement: Measurement,
    state: GaussianState,
    measure_model: LinearMeasurementModel,
) -> PosterioriResult:
    x = state.x
    P = state.P
    H = measure_model.H
    R = measure_model.R
    assert x.shape[0] == H.shape[
        1], "state and measurement model are not compatible"
    assert H.shape[0] == R.shape[0], "measurement model is not square"
    assert H.shape[0] == R.shape[1], "measurement model is not square"
    z = measurement
    inv = np.linalg.inv
    # innovation
    # the priori measurement residual
    y = z - H @ x
    # innovation covariance
    S = H @ P @ H.T + R
    # Kalman gain
    K = P @ H.T @ inv(S)
    # posteriori state
    x_posteriori = x + K @ y
    # dummy identity matrix
    I = np.eye(P.shape[0])
    # posteriori covariance
    I_KH = I - K @ H
    P_posteriori = I_KH @ P @ I_KH.T + K @ R @ K.T
    posteriori_state = GaussianState(x=x_posteriori, P=P_posteriori)
    posteriori_measurement = H @ x_posteriori
    s_m = y.T @ inv(S) @ y
    return PosterioriResult(
        state=posteriori_state,
        innovation=y,
        innovation_covariance=S,
        posteriori_measurement=posteriori_measurement,
        mahalanobis_distance=np.sqrt(s_m),
        squared_mahalanobis_distance=s_m,
    )


def cv_model(
    v_x: float,
    v_y: float,
    dt: float,
    q: float,
    r: float,
) -> Tuple[
        LinearMotionNoInputModel,
        LinearMeasurementModel,
        GaussianState,
]:
    """
    Create a constant velocity model with no input
    
    Args:
    v_x: initial velocity in x direction
    v_y: initial velocity in y direction
    dt: time interval
    q: process noise
    r: measurement noise

    Returns:
    motion_model: motion model
    measure_model: measurement model
    state: initial state
    """
    # yapf: disable
    F = np.array([[1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
    # yapf: enable
    Q = q * np.eye(4)
    R = r * np.eye(2)
    P = np.eye(4)
    motion_model = LinearMotionNoInputModel(F=F, Q=Q)
    measure_model = LinearMeasurementModel(H=H, R=R)
    state = GaussianState(x=np.array([0, 0, v_x, v_y]), P=P)
    return motion_model, measure_model, state


def outer_distance(x: NDArray, y: NDArray) -> NDArray:
    """
    Here's equivalent python code:
    
    ```python
    res = jnp.empty((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            # res[i, j] = jnp.linalg.norm(x[i] - y[j])
            res = res.at[i, j].set(jnp.linalg.norm(x[i] - y[j]))
    return res
    ```

    See Also
    --------
    `outer product <https://en.wikipedia.org/wiki/Outer_product>`_
    """

    x_expanded = x[:, None, :]
    y_expanded = y[None, :, :]
    diff = y_expanded - x_expanded
    return np.linalg.norm(diff, axis=-1)


@dataclass
class Tracking:
    id: int
    state: GaussianState
    survived_time_steps: int
    missed_time_steps: int


class TrackingEncoder(JSONEncoder):

    def default(self, o):
        if isinstance(o, Tracking):
            st = {
                "x": o.state.x.tolist(),
                "P": o.state.P.tolist(),
            }
            return {
                "id": o.id,
                "state": st,
                "survived_time_steps": o.survived_time_steps,
                "missed_time_steps": o.missed_time_steps,
            }
        return super().default(o)


@dataclass
class TrackerParams:
    dt: float = 1.0
    cov_threshold: float = 4.0
    tentative_mahalanobis_threshold: float = 10.0
    confirm_mahalanobis_threshold: float = 10.0
    forming_tracks_euclidean_threshold: float = 25.0
    survival_steps_threshold: int = 3


class Tracker:
    """
    A simple GNN tracker
    """
    _last_measurements: NDArray = np.empty((0, 2), dtype=np.float32)
    _tentative_tracks: list[Tracking] = []
    _confirmed_tracks: list[Tracking] = []
    _last_id: int = 0

    def __init__(self):
        self._last_measurements = np.empty((0, 2), dtype=np.float32)
        self._tentative_tracks = []
        self._confirmed_tracks = []

    @staticmethod
    def _predict(tracks: list[Tracking], dt: float = 1.0):
        return [
            Tracking(
                id=track.id,
                state=_predict(track.state, Tracker.motion_model(dt=dt)),
                survived_time_steps=track.survived_time_steps,
                missed_time_steps=track.missed_time_steps,
            ) for track in tracks
        ]

    @staticmethod
    def _data_associate_and_update(measurements: NDArray,
                                   tracks: list[Tracking],
                                   distance_threshold: float = 3) -> NDArray:
        """
        Match tracks with measurements and update the tracks

        Parameters
        ----------
        [in] measurements: Float["a 2"]
        [in,out] tracks: Tracking["b"]

        Returns
        ----------
        return 
            Float["... 2"] the unmatched measurements
        
        Effect
        ----------
        find the best match by minimum Mahalanobis distance, please note that I assume the state has been predicted
        """
        if len(tracks) == 0:
            return measurements

        def _update(measurement: NDArray, tracking: Tracking):
            return update(measurement, tracking.state,
                          Tracker.measurement_model())

        def outer_posteriori(
                measurements: NDArray,
                tracks: list[Tracking]) -> list[list[PosterioriResult]]:
            """
            calculate the outer posteriori for each measurement and track

            Parameters
            ----------
            [in] measurements: Float["a 2"]
            [in] tracks: Tracking["b"]

            Returns
            ----------
            PosterioriResult["a b"]
            """
            return [[
                _update(measurement, tracking) for measurement in measurements
            ] for tracking in tracks]

        def posteriori_to_mahalanobis(
                posteriori: list[list[PosterioriResult]]) -> NDArray:
            """
            Parameters
            ----------
            [in] posteriori: PosterioriResult["a b"]

            Returns
            ----------
            Float["a b"]
            """
            return np.array(
                [[r_m.mahalanobis_distance for r_m in p_t] for p_t in posteriori
                ],
                dtype=np.float32)

        posteriors = outer_posteriori(measurements, tracks)
        distances = posteriori_to_mahalanobis(posteriors)
        row, col = linear_sum_assignment(np.array(distances))
        row = np.array(row)
        col = np.array(col)

        def to_be_deleted() -> Generator[Tuple[int, int], None, None]:
            for i, j in zip(row, col):
                post: PosterioriResult = posteriors[i][j]
                if post.mahalanobis_distance > distance_threshold:
                    yield i, j

        for i, j in to_be_deleted():
            row = row[row != i]
            col = col[col != j]

        for i, j in zip(row, col):
            track: Tracking = tracks[i]
            post: PosterioriResult = posteriors[i][j]
            track.state = post.state
            track.survived_time_steps += 1
            tracks[i] = track

        for i, track in enumerate(tracks):
            if i not in row:
                # reset the survived time steps once missed
                track.missed_time_steps += 1
                tracks[i] = track
        # remove measurements that have been matched
        left_measurements = np.delete(measurements, col, axis=0)
        return left_measurements

    def _tracks_from_past_measurements(self,
                                       measurements: NDArray,
                                       dt: float = 1.0,
                                       distance_threshold: float = 3.0):
        """
        consume the last measurements and create tentative tracks from them

        Note
        ----
        mutate self._tentative_tracks and self._last_measurements
        """
        if self._last_measurements.shape[0] == 0:
            self._last_measurements = measurements
            return
        distances = outer_distance(self._last_measurements, measurements)
        row, col = linear_sum_assignment(distances)
        row = np.array(row)
        col = np.array(col)

        def to_be_deleted() -> Generator[Tuple[int, int], None, None]:
            for i, j in zip(row, col):
                euclidean_distance = distances[i, j]
                if euclidean_distance > distance_threshold:
                    yield i, j

        for i, j in to_be_deleted():
            row = row[row != i]
            col = col[col != j]

        for i, j in zip(row, col):
            coord = measurements[j]
            vel = (coord - self._last_measurements[i]) / dt
            s = np.concatenate([coord, vel])
            state = GaussianState(x=s, P=np.eye(4))
            track = Tracking(id=self._last_id,
                             state=state,
                             survived_time_steps=0,
                             missed_time_steps=0)
            self._last_id += 1
            self._tentative_tracks.append(track)
        # update the last measurements with the unmatched measurements
        self._last_measurements = np.delete(measurements, col, axis=0)

    def _transfer_tentative_to_confirmed(self,
                                         survival_steps_threshold: int = 3):
        """
        transfer tentative tracks to confirmed tracks

        Note
        ----
        mutate self._tentative_tracks and self._confirmed_tracks in place
        """
        for i, track in enumerate(self._tentative_tracks):
            if track.survived_time_steps > survival_steps_threshold:
                self._confirmed_tracks.append(track)
                self._tentative_tracks.pop(i)

    @staticmethod
    def _track_cov_deleter(tracks: list[Tracking], cov_threshold: float = 4.0):
        """
        delete tracks with covariance trace greater than threshold

        Parameters
        ----------
        [in,out] tracks: list[Tracking]
        cov_threshold: float
            the threshold of the covariance trace

        Note
        ----
        mutate tracks in place
        """
        for i, track in enumerate(tracks):
            # https://numpy.org/doc/stable/reference/generated/numpy.trace.html
            if np.trace(track.state.P) > cov_threshold:
                tracks.pop(i)

    def next_measurements(self, measurements: NDArray, params: TrackerParams):
        self._confirmed_tracks = self._predict(self._confirmed_tracks,
                                               params.dt)
        self._tentative_tracks = self._predict(self._tentative_tracks,
                                               params.dt)
        left_ = self._data_associate_and_update(
            measurements, self._confirmed_tracks,
            params.confirm_mahalanobis_threshold)
        left = self._data_associate_and_update(
            left_, self._tentative_tracks,
            params.tentative_mahalanobis_threshold)
        self._transfer_tentative_to_confirmed(params.survival_steps_threshold)
        self._tracks_from_past_measurements(
            left, params.dt, params.forming_tracks_euclidean_threshold)
        self._track_cov_deleter(self._tentative_tracks, params.cov_threshold)
        self._track_cov_deleter(self._confirmed_tracks, params.cov_threshold)

    @property
    def confirmed_tracks(self):
        return self._confirmed_tracks

    @staticmethod
    def motion_model(dt: float = 1,
                     q: float = 0.05) -> LinearMotionNoInputModel:
        """
        a constant velocity motion model
        """
        # yapf: disable
        F = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        # yapf: enable
        Q = q * np.eye(4)
        return LinearMotionNoInputModel(F=F, Q=Q)

    @staticmethod
    def measurement_model(r: float = 0.75) -> LinearMeasurementModel:
        # yapf: disable
        H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        # yapf: enable
        R = r * np.eye(2)
        return LinearMeasurementModel(H=H, R=R)


@dataclass
class CapProps:
    width: int
    height: int
    channels: int
    fps: float
    frame_count: Optional[int] = None


def fourcc(*args: str) -> int:
    return cv2.VideoWriter_fourcc(*args)  # type: ignore


def video_cap(
    src: PathLike | int | str,
    scale: float = 1,
) -> Tuple[Generator[MatLike, None, None], CapProps]:
    assert 0 < scale <= 1, "scale should be in (0, 1]"
    if isinstance(src, PathLike):
        cap = cv2.VideoCapture(str(src))
    else:
        cap = cv2.VideoCapture(src)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    channels = int(cap.get(cv2.CAP_PROP_CHANNEL))
    props = CapProps(width=width,
                     height=height,
                     fps=fps,
                     channels=channels,
                     frame_count=frame_count)

    def gen():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if scale != 1:
                frame = cv2.resize(frame, (width, height))
            yield frame
        cap.release()

    return gen(), props  # type: ignore


class DetectionFeatures(TypedDict):
    x: int
    y: int
    w: int
    h: int
    area: float
    cX: int
    cY: int


def main():
    tracker = Tracker()

    detections_history: list[list[DetectionFeatures]] = []
    tenative_histories: list[list[Tracking]] = []
    confirmed_histories: list[list[Tracking]] = []

    params = TrackerParams(
        cov_threshold=25.0,
        tentative_mahalanobis_threshold=50.0,
        confirm_mahalanobis_threshold=25.0,
        forming_tracks_euclidean_threshold=20,
        dt=1.0,
        survival_steps_threshold=6,
    )

    frames, props = video_cap("PETS09-S2L1-raw.mp4", 0.5)
    subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    logger.info("Video properties: {}", props)

    def process(frame: MatLike):
        fgmask = subtractor.apply(frame)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        dets: list[DetectionFeatures] = []
        detections: NDArray = np.empty((0, 2), dtype=np.float32)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area > 60:
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                features: DetectionFeatures = {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "area": area,
                    "cX": cX,
                    "cY": cY
                }
                dets.append(features)
                detections = np.vstack([detections, [cX, cY]])
        tracker.next_measurements(detections, params)
        detections_history.append(dets)
        tenative_histories.append(tracker._tentative_tracks.copy())
        confirmed_histories.append(tracker._confirmed_tracks.copy())

    try:
        for frame in tqdm(frames, total=props.frame_count):
            process(frame)
    except Exception as e:
        logger.exception(e)
    finally:
        with open("result.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "props": props.__dict__,
                    "detections_history": detections_history,
                    "confirmed_histories": confirmed_histories
                },
                f,
                cls=TrackingEncoder)


if __name__ == "__main__":
    main()
