feather.replace();

const minConfidence = 0.5;

const controls = document.querySelector('.controls');
const cameraOptions = document.querySelector('.video-options>select');
const video = document.querySelector('video');
const canvas = document.querySelector('canvas');
const screenshotImage = document.querySelector('img');
const buttons = [...controls.querySelectorAll('button')];
let streamStarted = false;
let imageCapture
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
  const track = stream.getVideoTracks()[0];
  const image = new ImageCapture(track);
  const photoBlob = await image.grabFrame();

  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  canvas.width = photoBlob.width;
  canvas.height = photoBlob.height;
  context.drawImage(photoBlob, 0, 0);

  const dataUrl = canvas.toDataURL(); // Convert to data URL
  return dataUrl; //Return promise
};

//Function for calling python api
async function predictImage(stream) {
  let frameURL;
  captureImage(stream).then((result) =>{
    console.log(result);
    frameURL = result;
  });

  const response = await fetch('http://127.0.0.1:5000/predictImage', {
      method: 'POST',
      body: frameURL
  });
  return response;
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

  predictImage(stream).then((result) => {
    console.log(result);
  });

  streamStarted = true;
};

getCameraSelection();