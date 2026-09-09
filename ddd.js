import { FilesetResolver, FaceLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

// DDD uses OpenCV.js for frame-quality analysis and MediaPipe for robust eye landmarks.
const MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const LEFT = [33, 160, 158, 133, 153, 144], RIGHT = [362, 385, 387, 263, 373, 380];
const CALIBRATION_FRAMES = 75, CLOSED_SECONDS = 1.5, THRESHOLD_RATIO = .72;
const $ = id => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), ctx = overlay.getContext("2d");
const work = document.createElement("canvas"), workCtx = work.getContext("2d", { willReadFrequently: true });
let cvReady, landmarker, stream, running = false, raf = 0, lastVideoTime = -1;
let leftSamples = [], rightSamples = [], baseL = 0, baseR = 0, smoothL, smoothR, closedAt = null, alarmTimer, audio;

function loadCV() {
  if (cvReady) return cvReady;
  cvReady = new Promise((resolve, reject) => {
    window.Module = { onRuntimeInitialized: resolve };
    const script = document.createElement("script"); script.src = "opencv.js"; script.onerror = () => reject(new Error("OpenCV.js could not load. Keep opencv.js beside ddd.html.")); document.head.append(script);
  });
  return cvReady;
}
const distance = (a,b) => Math.hypot(a.x-b.x, a.y-b.y);
function ear(points) { const width = distance(points[0],points[3]); return width ? (distance(points[1],points[5])+distance(points[2],points[4]))/(2*width) : 0; }
function median(a) { const s=[...a].sort((x,y)=>x-y), cut=Math.floor(s.length*.12), t=s.slice(cut,s.length-cut); return t.reduce((x,y)=>x+y,0)/t.length; }
function resetCalibration() { leftSamples=[]; rightSamples=[]; baseL=baseR=0; smoothL=smoothR=undefined; closedAt=null; stopAlarm(); $("closed").textContent="0.00 s"; }
function beep() { audio ||= new (window.AudioContext||window.webkitAudioContext)(); const o=audio.createOscillator(), g=audio.createGain(); g.gain.value=.12; o.frequency.value=900; o.connect(g).connect(audio.destination); o.start(); o.stop(audio.currentTime+.22); }
function startAlarm() { if (!alarmTimer) { beep(); alarmTimer=setInterval(beep,550); } }
function stopAlarm() { if (alarmTimer) { clearInterval(alarmTimer); alarmTimer=null; } }

function frameQuality() {
  // Real OpenCV processing: grayscale mean (exposure) and Laplacian variance (focus).
  workCtx.drawImage(video,0,0,work.width,work.height);
  const rgba=cv.imread(work), gray=new cv.Mat(), lap=new cv.Mat(), mean=cv.mean(rgba)[0];
  cv.cvtColor(rgba,gray,cv.COLOR_RGBA2GRAY); cv.Laplacian(gray,lap,cv.CV_64F);
  const sharpness=cv.mean(lap)[0]; // a cheap, mobile-friendly blur estimate
  rgba.delete(); gray.delete(); lap.delete();
  const usable=mean>35 && mean<235 && sharpness>5;
  $("quality").textContent=usable ? `Good (${Math.round(mean)})` : `Improve light/focus (${Math.round(mean)})`;
  return usable;
}
function draw(points) { ctx.clearRect(0,0,overlay.width,overlay.height); ctx.save(); ctx.translate(overlay.width,0); ctx.scale(-1,1); ctx.strokeStyle="#4ce07a"; ctx.fillStyle="#4ce07a"; ctx.lineWidth=2; for (const index of [...LEFT,...RIGHT]) { const p=points[index]; ctx.beginPath(); ctx.arc(p.x*overlay.width,p.y*overlay.height,3,0,Math.PI*2); ctx.fill(); } ctx.restore(); }
function setStatus(text, alert=false) { $("status").textContent=text; $("status").classList.toggle("alert",alert); }
function process(landmarks, now, usable) {
  draw(landmarks); $("face").textContent="Detected";
  const l=ear(LEFT.map(i=>landmarks[i])), r=ear(RIGHT.map(i=>landmarks[i]));
  smoothL=smoothL===undefined?l:.72*smoothL+.28*l; smoothR=smoothR===undefined?r:.72*smoothR+.28*r;
  if (leftSamples.length<CALIBRATION_FRAMES) { leftSamples.push(smoothL); rightSamples.push(smoothR); $("eyes").textContent=`Calibrating ${leftSamples.length}/${CALIBRATION_FRAMES}`; setStatus("Calibrating — look forward with both eyes open"); if(leftSamples.length===CALIBRATION_FRAMES) { baseL=median(leftSamples)*THRESHOLD_RATIO; baseR=median(rightSamples)*THRESHOLD_RATIO; } return; }
  if (!usable) { closedAt=null; stopAlarm(); $("eyes").textContent="Quality pause"; setStatus("Improve lighting or camera focus"); return; }
  const shut=smoothL<baseL && smoothR<baseR;
  if (!shut) { closedAt=null; stopAlarm(); $("closed").textContent="0.00 s"; $("eyes").textContent="Eyes open"; setStatus("Monitoring"); return; }
  closedAt ??= now; const seconds=(now-closedAt)/1000; $("closed").textContent=`${seconds.toFixed(2)} s`;
  if(seconds>=CLOSED_SECONDS) { $("eyes").textContent="DROWSINESS ALERT"; setStatus("DROWSINESS ALERT — take a break",true); startAlarm(); } else { $("eyes").textContent="Eyes closed"; setStatus("Eyes closed"); }
}
function loop(now) { if(!running) return; if(video.readyState>=2 && video.currentTime!==lastVideoTime) { lastVideoTime=video.currentTime; try { const usable=frameQuality(), result=landmarker.detectForVideo(video,Math.round(now)); if(result.faceLandmarks?.[0]) process(result.faceLandmarks[0],now,usable); else { ctx.clearRect(0,0,overlay.width,overlay.height); $("face").textContent="No face"; $("eyes").textContent="—"; closedAt=null; stopAlarm(); setStatus("Position your face in the camera"); } } catch(error) { $("error").textContent=error.message; } } raf=requestAnimationFrame(loop); }
async function start() { try { $("error").textContent=""; setStatus("Loading models…"); await Promise.all([loadCV(), (async()=>{ const vision=await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm"); landmarker=await FaceLandmarker.createFromOptions(vision,{baseOptions:{modelAssetPath:MODEL,delegate:"CPU"},runningMode:"VIDEO",numFaces:1,minFaceDetectionConfidence:.6,minFacePresenceConfidence:.6,minTrackingConfidence:.6}); })()]); stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"user",width:{ideal:960},height:{ideal:540}},audio:false}); video.srcObject=stream; await video.play(); overlay.width=work.width=video.videoWidth; overlay.height=work.height=video.videoHeight; resetCalibration(); running=true; $("hint").style.display="none"; $("start").disabled=true; $("stop").disabled=false; $("recalibrate").disabled=false; raf=requestAnimationFrame(loop); } catch(error) { setStatus("Could not start",true); $("error").textContent=error.message+" GitHub Pages must be opened over HTTPS."; } }
function stop() { running=false; cancelAnimationFrame(raf); stopAlarm(); stream?.getTracks().forEach(t=>t.stop()); stream=null; video.srcObject=null; $("start").disabled=false; $("stop").disabled=true; $("recalibrate").disabled=true; $("hint").style.display="grid"; setStatus("Stopped"); }
$("start").onclick=start; $("stop").onclick=stop; $("recalibrate").onclick=resetCalibration; addEventListener("beforeunload",stop);
