import { FilesetResolver, FaceLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const LEFT = [33, 160, 158, 133, 153, 144];
const RIGHT = [362, 385, 387, 263, 373, 380];
const SAMPLES = 75, CLOSED_SECONDS = 1.5, THRESHOLD_RATIO = .72;
const $ = id => document.getElementById(id);
const video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");
const work = document.createElement("canvas"), workCtx = work.getContext("2d", { willReadFrequently: true });
let cvPromise, landmarker, stream, running = false, animation, lastVideoTime = -1;
let leftValues = [], rightValues = [], thresholdLeft = 0, thresholdRight = 0, smoothLeft, smoothRight, closedAt = null, alarm, audio;
let lastQualityCheck = 0, latestQuality = true;

function loadOpenCV() {
  if (cvPromise) return cvPromise;
  cvPromise = new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("OpenCV.js took too long to load. Check that opencv.js is in the GitHub Pages folder.")), 30000);
    const script = document.createElement("script");
    script.src = "opencv.js";
    script.onerror = () => { clearTimeout(timeout); reject(new Error("OpenCV.js was not found. Upload opencv.js beside ddd.html.")); };
    script.onload = () => {
      clearTimeout(timeout);
      if (window.cv?.Mat) resolve(window.cv);
      else reject(new Error("OpenCV.js loaded but did not expose its API. Replace the uploaded opencv.js with the copy in this project folder."));
    };
    document.head.append(script);
  });
  return cvPromise;
}

const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
function eyeRatio(points) { const width = distance(points[0], points[3]); return width ? (distance(points[1], points[5]) + distance(points[2], points[4])) / (2 * width) : 0; }
function trimmedMean(values) { const sorted = [...values].sort((a,b) => a-b), cut = Math.floor(values.length * .12), kept = sorted.slice(cut, sorted.length-cut); return kept.reduce((sum,value) => sum+value,0) / kept.length; }
function setStatus(text, warning = false) { $("status").textContent = text; $("status").style.color = warning ? "#ff4b54" : ""; }
function resetCalibration() { leftValues=[]; rightValues=[]; thresholdLeft=thresholdRight=0; smoothLeft=smoothRight=undefined; closedAt=null; stopAlarm(); $("leftEar").textContent=$("rightEar").textContent=$("threshold").textContent="--"; $("closed").textContent="0.00 s"; }
function stopAlarm() { if (alarm) { clearInterval(alarm); alarm=null; } }
function beep() { audio ||= new (window.AudioContext || window.webkitAudioContext)(); const oscillator=audio.createOscillator(), gain=audio.createGain(); gain.gain.value=.12; oscillator.frequency.value=900; oscillator.connect(gain).connect(audio.destination); oscillator.start(); oscillator.stop(audio.currentTime+.22); }
function startAlarm() { if (!alarm) { beep(); alarm=setInterval(beep,550); } }

function getOpenCVQuality() {
  const CV = window.cv;
  workCtx.drawImage(video, 0, 0, work.width, work.height);
  const rgba = CV.imread(work), gray = new CV.Mat(), laplacian = new CV.Mat(), mean = new CV.Mat(), deviation = new CV.Mat();
  const brightness = CV.mean(rgba)[0];
  CV.cvtColor(rgba, gray, CV.COLOR_RGBA2GRAY);
  CV.Laplacian(gray, laplacian, CV.CV_64F);
  CV.meanStdDev(laplacian, mean, deviation);
  const sharpness = deviation.doubleAt(0, 0) ** 2; // Laplacian variance: a focus estimate
  rgba.delete(); gray.delete(); laplacian.delete(); mean.delete(); deviation.delete();
  const good = brightness > 35 && brightness < 235 && sharpness > 5;
  $("quality").textContent = good ? `GOOD (${Math.round(brightness)})` : `CHECK LIGHT (${Math.round(brightness)})`;
  return good;
}

function qualityForFrame(now) {
  // OpenCV runs on a small image once per second. Running it on every 960×540
  // camera frame can make mobile browsers unresponsive.
  if (now - lastQualityCheck > 1000) {
    latestQuality = getOpenCVQuality();
    lastQualityCheck = now;
  }
  return latestQuality;
}

function drawFace(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const points = landmarks.map(p => ({ x:p.x*canvas.width, y:p.y*canvas.height }));
  const xs=points.map(p=>p.x), ys=points.map(p=>p.y);
  ctx.save(); ctx.strokeStyle=ctx.fillStyle="#3cc850"; ctx.lineWidth=3;
  ctx.strokeRect(Math.min(...xs), Math.min(...ys), Math.max(...xs)-Math.min(...xs), Math.max(...ys)-Math.min(...ys));
  for (const index of [...LEFT,...RIGHT]) { const p=points[index]; ctx.beginPath(); ctx.arc(p.x,p.y,4,0,Math.PI*2); ctx.fill(); }
  ctx.restore();
}

function processFace(landmarks, now, qualityIsGood) {
  drawFace(landmarks); $("face").textContent="DETECTED";
  const left=eyeRatio(LEFT.map(i=>landmarks[i])), right=eyeRatio(RIGHT.map(i=>landmarks[i]));
  smoothLeft=smoothLeft === undefined ? left : .72*smoothLeft+.28*left;
  smoothRight=smoothRight === undefined ? right : .72*smoothRight+.28*right;
  $("leftEar").textContent=smoothLeft.toFixed(3); $("rightEar").textContent=smoothRight.toFixed(3);
  if (leftValues.length < SAMPLES) {
    leftValues.push(smoothLeft); rightValues.push(smoothRight);
    $("eyes").textContent=`CALIBRATING ${leftValues.length}/${SAMPLES}`; setStatus("CALIBRATING - KEEP EYES OPEN");
    if (leftValues.length === SAMPLES) { thresholdLeft=trimmedMean(leftValues)*THRESHOLD_RATIO; thresholdRight=trimmedMean(rightValues)*THRESHOLD_RATIO; $("threshold").textContent=`${thresholdLeft.toFixed(3)} / ${thresholdRight.toFixed(3)}`; }
    return;
  }
  if (!qualityIsGood) { closedAt=null; stopAlarm(); $("eyes").textContent="QUALITY PAUSE"; setStatus("IMPROVE LIGHTING OR CAMERA FOCUS"); return; }
  const bothClosed = smoothLeft < thresholdLeft && smoothRight < thresholdRight;
  if (!bothClosed) { closedAt=null; stopAlarm(); $("closed").textContent="0.00 s"; $("eyes").textContent="EYES OPEN"; setStatus("MONITORING"); return; }
  closedAt ??= now; const seconds=(now-closedAt)/1000; $("closed").textContent=`${seconds.toFixed(2)} s`;
  if (seconds >= CLOSED_SECONDS) { $("eyes").textContent="DROWSINESS ALERT!"; setStatus("DROWSINESS ALERT - TAKE A BREAK",true); startAlarm(); }
  else { $("eyes").textContent="EYES CLOSED"; setStatus("EYES CLOSED"); }
}

function loop(now) {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime=video.currentTime;
    try { const quality=qualityForFrame(now), result=landmarker.detectForVideo(video,Math.round(now)); if (result.faceLandmarks?.[0]) processFace(result.faceLandmarks[0],now,quality); else { ctx.clearRect(0,0,canvas.width,canvas.height); $("face").textContent="NO FACE"; $("eyes").textContent="NO FACE"; closedAt=null; stopAlarm(); setStatus("POSITION YOUR FACE IN THE CAMERA"); } }
    catch (error) { $("error").textContent=`Detection error: ${error.message}`; }
  }
  animation=requestAnimationFrame(loop);
}

async function start() {
  let startupStep = "face model";
  try {
    $("error").textContent=""; setStatus("LOADING FACE MODEL...");
    // MediaPipe must finish first. Starting OpenCV at the same time can lock up
    // the browser while both WebAssembly runtimes are initializing.
    if (!landmarker) {
      const vision=await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm");
      landmarker=await FaceLandmarker.createFromOptions(vision,{baseOptions:{modelAssetPath:MODEL,delegate:"CPU"},runningMode:"VIDEO",numFaces:1,minFaceDetectionConfidence:.6,minFacePresenceConfidence:.6,minTrackingConfidence:.6});
    }
    startupStep = "OpenCV"; setStatus("LOADING OPENCV...");
    await loadOpenCV();
    startupStep = "camera permission";
    stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"user",width:{ideal:960},height:{ideal:540}},audio:false}); video.srcObject=stream; await video.play();
    canvas.width=video.videoWidth; canvas.height=video.videoHeight;
    // A low-resolution copy is enough for brightness/focus evaluation.
    work.width=160; work.height=90; lastQualityCheck=0; latestQuality=true;
    resetCalibration(); running=true; $("message").style.display="none"; $("start").disabled=true; $("stop").disabled=false; $("recalibrate").disabled=false; animation=requestAnimationFrame(loop);
  } catch (error) { console.error(`DDD startup failed during ${startupStep}:`, error); setStatus("COULD NOT START",true); $("error").textContent=`Failed during ${startupStep}: ${error.message} Use GitHub Pages (HTTPS) and allow camera access.`; }
}
function stop() { running=false; cancelAnimationFrame(animation); stopAlarm(); stream?.getTracks().forEach(track=>track.stop()); stream=null; video.srcObject=null; $("start").disabled=false; $("stop").disabled=true; $("recalibrate").disabled=true; $("message").style.display="block"; setStatus("STOPPED"); }
$("start").onclick=start; $("stop").onclick=stop; $("recalibrate").onclick=resetCalibration; addEventListener("beforeunload",stop);
