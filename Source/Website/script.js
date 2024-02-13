feather.replace();

const minConfidence = 0.5;

const labelOutput = document.getElementById("label-output");
const confidenceOutput = document.getElementById("confidence-output");
const inferenceTimeOutput = document.getElementById("inference-time-output");

const controls = document.querySelector('.controls');
const cameraOptions = document.querySelector('.video-options>select');
const video = document.querySelector('video');
const canvas = document.querySelector('canvas');
const screenshotImage = document.querySelector('img');
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

async function captureImage(stream){
  //Get track from stream then get frame from track
  const track = stream.getVideoTracks()[0];
  const image = new ImageCapture(track);
  const photoBlob = await image.grabFrame();

  //Add frame to canvas so it could be convert to data URL later
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  canvas.width = 192; //Resize image to decrease load on the server
  canvas.height = 144;
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

async function APIgetTest() {
  const response = await fetch('http://127.0.0.1:5000/APIgetTest', {
    method: 'GET',
    headers: {
        'Content-Type': 'application/json'
    }
  });
  return response.json();
};

async function APIpostTest(inputData) {
  const response = await fetch('http://127.0.0.1:5000/APIpostTest', {
      method: 'POST',
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(inputData)
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

  //APIgetTest().then((result) => {
  //  console.log(result);
  //});
  //APIpostTest({"lel" : "sheet"}).then((result) => {
  //  console.log(result);
  //});
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
      if (result["label"] != null) {
        console.log(result);
        //const node = document.createTextNode(result["label"])
        labelOutput.innerHTML = "Output: " + result["label"];
        confidenceOutput.innerHTML = "Confidence: " + result["confidence"] + "%";
        inferenceTimeOutput.innerHTML = "Inference time: " + result["inferenceTime"] + "s";
      }
    });
  };
  setInterval(getPrediction, 100, stream)//Interval that with predict label of current frame!
};

getCameraSelection();