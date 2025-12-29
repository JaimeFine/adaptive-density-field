# **Physics-informed Trajectory POI Detection Pipeline**

**Dated: December 29, 2025**

---

## 1 Preprocessing the Flight Data

---

### 1.1 Coordinate Conversion: WGS84 Geodetic to ECEF

Given:

- latitude $\varphi$ (rad)  
- longitude $\lambda$ (rad)  
- ellipsoidal height $h$ (m)  
- WGS84 parameters:  
  - semi‑major axis $a = 6378137.0$ 
  - flattening rate $f = \frac{1}{298.257223563}$
  
  - first eccentricity squared $e^2 = 6.69437999014\times 10^{-3}$

First compute the prime vertical radius of curvature:

$$
N(\varphi) = \frac{a}{\sqrt{1 - e^2 \sin^2\varphi}} \tag{1.1.1}
$$

Then ECEF coordinates $(x,y,z)$:

$
\begin{equation}
\tag{1.1.2}
\begin{aligned}
x &= \left(N(\varphi) + h\right)\cos\varphi\cos\lambda \\
y &= \left(N(\varphi) + h\right)\cos\varphi\sin\lambda \\
z &= \left(N(\varphi)(1 - e^2) + h\right)\sin\varphi
\end{aligned}
\end{equation}
$

Hence we get the ENU coordinates.

---

### 1.2 Coordinate Conversion: ECEF to ENU Conversion

Pick a reference point (the origin of the local ENU frame in this case) with geodetic coordinates $(\varphi_0,\lambda_0,h_0)$, and compute its ECEF coordinates $(x_0,y_0,z_0)$ using the same equations as above.

For any point with ECEF $(x,y,z)$, define the difference vector:

$
\begin{equation}
\tag{1.2.1}
\begin{bmatrix}
\Delta x \\ \Delta y \\ \Delta z
\end{bmatrix}=\begin{bmatrix}
x - x_0 \\ y - y_0 \\ z - z_0
\end{bmatrix}
\end{equation}
$

And given the Rotation matrix and ENU coordinate at reference $(\varphi_0,\lambda_0,h_0)$:

$
\begin{equation}
\tag{1.2.2}
\mathbf{R}=\begin{bmatrix}
\sin\varphi_0 & \cos\varphi_0 & 0 \\
\cos\varphi_0\cdot\sin\lambda_0 & -\sin\varphi_0\cdot\sin\lambda_0 & \cos\lambda_0 \\
\cos\varphi_0\cdot\cos\lambda_0 & \sin\varphi_0\cdot\cos\lambda_0 & \sin\lambda_0
\end{bmatrix}
\end{equation}
$

Therefore we have the calculation:

$
\begin{equation}
\tag{1.2.3}
\begin{bmatrix}
E \\ N \\ U
\end{bmatrix}=\begin{bmatrix}
\Delta x \\ \Delta y \\ \Delta z
\end{bmatrix}\cdot\mathbf{R}
\end{equation}
$

This is the standard ECEF → ENU transformation used in geodesy and navigation.

---

### 1.3 Creating a Dictionary

To organize per‑flight data extracted from each GeoJSON file, we build a dictionary where each flight ID maps to three lists:

- coords — longitude, latitude, altitude
- vel — velocity components 
- dt — timestamps

The basic structure looks like this:

```txt
    flights = dict({
        "coords": [],
        "vel": [],
        "dt": []
    })
```

In practice, we use a defaultdict so each new flight_id automatically initializes this structure.

---

## 2 Position Prediction

To estimate future aircraft positions, I applied a **physics‑based interpolation model** that blends two motion predictors:

1. **Constant‑Acceleration (CA) model** — reliable for nearly straight trajectories  
2. **Cubic Hermite Spline interpolation** — smooth and accurate for curved motion  

The blending weight is determined by the **local curvature** of the trajectory:
- Low curvature → motion is nearly straight → CA dominates  
- High curvature → motion bends → spline dominates

This adaptive combination produces a more stable and realistic prediction than using either method alone.

---

### 2.1 General Prediction

For each flight:

- Convert raw coordinates into a consistent Cartesian frame  
- Compute velocity and approximate acceleration  
- Estimate local curvature $k$ using  
  $
  \begin{equation}
  \tag{2.1.1}
  k = \frac{\lVert \mathbf{v} \times \mathbf{a} \rVert}{\lVert \mathbf{v} \rVert^3}
  \end{equation}
  $
- Compute a flight‑specific smoothing parameter  
  $
  \begin{equation}
  \tag{2.1.2}
  \alpha = \frac{\ln 5}{k_{95}}
  \end{equation}
  $
  where $k_{95}$ is the 95th percentile curvature  
- For each timestamp, compute:
  - **Spline prediction** using `CubicHermiteSpline`
  - **Constant‑acceleration prediction**
- Blend them using $w = e^{-\alpha k}$:
  $
  \begin{equation}
  \tag{2.1.3}
  \hat{p} = w \, p_{\text{CA}} + (1 - w) \, p_{\text{spline}}
  \end{equation}
  $

This yields a smooth, curvature‑aware prediction for each flight.

---

### 2.2 Loss Computation

To evaluate the quality of the predicted positions, I compute a **time‑normalized Mahalanobis loss** for each flight. This metric captures not only the magnitude of prediction errors but also their **directional structure**, **covariance**, and **temporal spacing**.

The loss is computed in four main steps:

**1. Extract Prediction Residuals**

For each flight, I compare the predicted positions $\hat{p}_i$ with the actual converted coordinates $p_i$:

$
\begin{equation}
\tag{2.2.1}
r_i = \hat{p}_i - p_i
\end{equation}
$

Only interior points are used `[2 : size-2]` to avoid boundary artifacts from the spline and acceleration models.

The residuals are then centered:

$
\begin{equation}
\tag{2.2.2}
\tilde{r}_i = r_i - \bar{r}
\end{equation}
$

This removes global bias and ensures the covariance reflects *shape* rather than offset.

**2. Estimate Residual Covariance**

The covariance of the centered residuals is computed as:

$
\begin{equation}
\tag{2.2.3}
\Sigma = \operatorname{Cov}(\tilde{r}) + \lambda I
\end{equation}
$

A small Tikhonov regularization term $\lambda = 10^{-5}$ stabilizes the inversion of $\Sigma$, especially for nearly collinear trajectories.

The inverse covariance $\Sigma^{-1}$ defines the **Mahalanobis geometry** of the error space.

**3. Compute Mahalanobis Distance**

For each residual vector:

$
\begin{equation}
\tag{2.2.4}
d_i = \sqrt{\, \tilde{r}_i^\top \Sigma^{-1} \tilde{r}_i \,}
\end{equation}
$

This distance penalizes errors more strongly along directions where the model is normally precise, and less along directions with naturally higher variance.

**4. Normalize by Temporal Spacing**

Because timestamps are not uniformly spaced, each error is scaled by a time‑dependent factor:

$
\begin{equation}
\tag{2.2.5}
t_i = \sqrt{\frac{\Delta t_i}{\bar{\Delta t}}}
\end{equation}
$

The final **time-relative Mahalanobis loss** is:

$
\begin{equation}
\tag{2.2.6}
L_i = \frac{d_i}{t_i}
\end{equation}
$

This ensures that predictions made over longer time intervals are not unfairly penalized compared to short‑interval predictions.

## 3 POI Detection

After computing the time‑normalized Mahalanobis loss for each flight, the next step is to identify **Points of Interest (POIs)**—locations where the prediction error is unusually high. These points often correspond to sharp maneuvers, abnormal motion, or sensor irregularities, and they serve as valuable markers for downstream analysis.

**However, a POI does not always represent an actual infrastructure feature; it simply marks a point where the motion deviates significantly.**

The POI detection pipeline consists of three main stages:

---

### 3.1 Normalize the Loss Scores

For each flight, the Mahalanobis losses are rescaled to the interval `[0, 1]`:

$
\begin{equation}
\tag{3.1}
s_i = \frac{L_i - \min(L)}{\max(L) - \min(L) + \varepsilon}
\end{equation}
$

This normalization ensures that POI detection is **relative to each flight’s own dynamics**, making the method robust to differences in scale, speed, or noise across flights.

---

### 3.2 Thresholding

Here, I introduced an element called POI score, which indicates how anomalous each point is relative to the rest of the flight.

A point is flagged as a POI if its normalized score exceeds a fixed threshold:

$
\begin{equation}
\tag{3.2}
s_i \ge 0.75
\end{equation}
$

This threshold captures the upper quartile of anomalous behavior while avoiding excessive false positives.

It can be adjusted depending on the desired sensitivity of the detection process.

---

### 3.3 Export POIs to CSV

Each detected POI is stored with:

- flight ID  
- point index  
- longitude, latitude, altitude  
- POI score  

All POIs are aggregated into a Pandas DataFrame and exported as a CSV file, enabling further visualization, inspection, or integration into downstream workflows.