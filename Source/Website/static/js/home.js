feather.replace();

const minConfidence = 0.5;
const fps = 24;
const resizeRatio = 10;

const labelOutput = document.getElementById("label-output");
const confidenceOutput = document.getElementById("confidence-output");
const inferenceTimeOutput = document.getElementById("inference-time-output");

const optionDiv = document.getElementById("setting-box");

const controls = document.querySelector('.controls');
const cameraOptions = document.querySelector('.video-options>select');
const video = document.querySelector('video');
const canvas = document.querySelector('canvas');
const buttons = [...controls.querySelectorAll('button')];
let streamStarted = false;
const [play, pause, screenshot] = buttons;

const constraints = {
  video: {
    width: {
      min: 1280,
      ideal: 1920,
      max: 2560,
    },
    height: {
      min: 720,
      ideal: 1080,
      max: 1440
    },
  }
};


function showSetting() {
  console.log("executed");
  optionDiv.style.display = "block";
};

function closeSetting() {
  console.log("executed");
  optionDiv.style.display = "none";
};


async function captureImage(stream){
  //Get track from stream then get frame from track
  const track = stream.getVideoTracks()[0];
  const image = new ImageCapture(track);
  const photoBlob = await image.grabFrame();

  //Add frame to canvas so it could be convert to data URL later
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  canvas.width = 1920 / resizeRatio; //Resize image to decrease load on the server
  canvas.height = 1080 / resizeRatio;
  context.drawImage(photoBlob, 0, 0, canvas.width, canvas.height);

  const dataUrl = canvas.toDataURL(); // Convert to data URL
  return dataUrl; //Return promise
};

//Function for calling python api
async function callPredictImage(stream) {
  let frameURL = await captureImage(stream);
  const response = await fetch('http://127.0.0.1:5000/predictImage', {
      method: 'POST',
      body: frameURL
  });
  return response.json();
};

const getCameraSelection = async () => {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoDevices = devices.filter(device => device.kind === 'videoinput');
  const options = videoDevices.map(videoDevice => {
    return `<option value="${videoDevice.deviceId}">${videoDevice.label}</option>`;
  });
  cameraOptions.innerHTML = options.join('');
};

play.onclick = () => {
  if (streamStarted) {
    video.play();
    play.classList.add('d-none');
    pause.classList.remove('d-none');
    return;
  }
  if ('mediaDevices' in navigator && navigator.mediaDevices.getUserMedia) {
    const updatedConstraints = {
      ...constraints,
      deviceId: {
        exact: cameraOptions.value
      }
    };
    startStream(updatedConstraints);
  }
};

const startStream = async (constraints) => {
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  handleStream(stream);
};

const handleStream = (stream) => {
  video.srcObject = stream;
  play.classList.add('d-none');
  pause.classList.remove('d-none');
  screenshot.classList.remove('d-none');

  streamStarted = true;

  //Set up function so it could be use with interval
  function getPrediction(stream) {
    callPredictImage(stream).then((result) => {
      if (result != null) {
        if (result["confidence"] > minConfidence * 100) {
          console.log(result);
          //const node = document.createTextNode(result["label"])
          labelOutput.innerHTML = "Output: " + result["label"];
          confidenceOutput.innerHTML = "Confidence: " + result["confidence"] + "%";
          inferenceTimeOutput.innerHTML = "Inference time: " + result["inferenceTime"] + "s";
        };
      };
    });
  };
  setInterval(getPrediction, 1000 / fps, stream)//Interval that with predict label of current frame!
};

getCameraSelection();