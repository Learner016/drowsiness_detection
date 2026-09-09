# Driver Drowsiness Detection – Final Web Version

A simple browser-based Computer Vision project using MediaPipe Face Landmarker and Eye Aspect Ratio (EAR).

## Windows

1. Install Python 3 if it is not already installed.
2. Extract this folder.
3. Double-click **RUN.bat**.
4. The browser opens at `http://localhost:8000/index.html`.
5. Click **START CAMERA** and allow camera permission.
6. Keep both eyes open during automatic calibration.

Do **not** open `index.html` directly. Use `RUN.bat` so the camera runs from localhost.

## Android

Yes. This web version can run on Android in a modern browser such as Chrome, provided the site is opened over **HTTPS**.

### Recommended Android method: GitHub Pages

1. Create a GitHub repository.
2. Upload `index.html`, `style.css`, and `app.js`.
3. Enable **GitHub Pages** for the repository.
4. Open the HTTPS Pages address on the Android phone.
5. Allow camera permission.
6. Press **START CAMERA**.

You do not need Python, Pydroid, OpenCV installation, or a separate Android app for the web version.

## OpenCV-assisted version (`ddd.html`)

`ddd.html` is a separate, mobile-friendly version. It uses OpenCV.js to check
lighting and focus before allowing an eye-closure alert, and uses MediaPipe Face
Landmarker for the eye landmarks. Open `ddd.html` from GitHub Pages after
uploading these three files together:

- `ddd.html`
- `ddd.js`
- `opencv.js`

For example, if your Pages URL is `https://USERNAME.github.io/REPOSITORY/`, open
`https://USERNAME.github.io/REPOSITORY/ddd.html`. Camera access requires HTTPS
(which GitHub Pages provides) and the user must allow the browser camera prompt.

## Alarm behavior

- Both eyes closed for less than 1.5 seconds: **BOTH EYES CLOSED**.
- Both eyes continuously closed for 1.5 seconds: **DROWSINESS ALERT!**.
- The alarm then repeats a beep approximately every 0.6 seconds while both eyes remain closed.
- The alarm stops immediately when the eyes open, one eye opens/closes, the face disappears, or the camera is stopped.
- One continuous eye-closure event counts as one alert.
- Starting the camera again resets the alert count, timer, calibration, and alarm state.

## Detection

- One face at a time.
- Six landmarks per eye.
- Roll-normalized EAR for improved stability when the head is rotated sideways.
- 60-frame automatic calibration.
- Per-eye threshold = calibrated EAR × 0.72.
- EAR smoothing factor = 0.7.
- 1.5-second continuous eye-closure alert.
- Browser Web Audio alarm.

MediaPipe Face Landmarker supports web JavaScript applications and VIDEO mode for camera/video frames. See Google's official documentation for details.

## Privacy

Camera frames are processed locally in the browser. This project does not upload camera frames to a server.

## Important

This is an academic Computer Vision project and is not a certified vehicle safety system.
