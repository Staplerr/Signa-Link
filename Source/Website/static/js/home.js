feather.replace();

const minConfidence = 0.5;
const fps = 30;
const resizeRatio = 5;
const maxNone = 150;
let noneCount = 0;

const labelOutput = document.getElementById("label-output");
const confidenceOutput = document.getElementById("confidence-output");
const inferenceTimeOutput = document.getElementById("inference-time-output");

const settingBox = document.getElementById("setting-box");

const controls = document.querySelector(".controls");
const cameraOptions = document.querySelector(".video-options>select");
const video = document.querySelector("video");
const canvas = document.querySelector("canvas");
const buttons = [...controls.querySelectorAll("button")];
let streamStarted = false;
const [play, pause, screenshot] = buttons;

const constraints = {
  video: {
    width: {
      min: 1280,
      ideal: 1280,
      max: 1920,
    },
    height: {
      min: 720,
      ideal: 720,
      max: 1080,
    },
  },
};

function showSetting() {
  settingBox.style.display = "block";
}

function closeSetting() {
  settingBox.style.display = "none";
}

async function captureImage(stream) {
  // Get track from stream then get frame from track
  const track = stream.getVideoTracks()[0];
  const imageCapture = new ImageCapture(track);
  const photoBlob = await imageCapture.grabFrame();

  // Add frame to canvas so it could be converted to data URL later
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  canvas.width = constraints.video.width.ideal / resizeRatio; // Resize image to decrease load on the server
  canvas.height = constraints.video.height.ideal / resizeRatio;
  context.drawImage(photoBlob, 0, 0, canvas.width, canvas.height);

  const dataUrl = canvas.toDataURL(); // Convert to data URL
  return dataUrl; // Return promise
}

// Function for calling python API
async function callPredictImage(stream) {
  let frameURL = await captureImage(stream);
  const response = await fetch("http://127.0.0.1:5000/predictImage", {
    method: "POST",
    body: frameURL,
  });
  return response.json();
}

const getCameraSelection = async () => {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoDevices = devices.filter((device) => device.kind === "videoinput");
  const options = videoDevices.map((videoDevice) => {
    return `<option value="${videoDevice.deviceId}">${videoDevice.label}</option>`;
  });
  cameraOptions.innerHTML = options.join("");
};

play.onclick = () => {
  if (streamStarted) {
    video.play();
    play.classList.add("d-none");
    pause.classList.remove("d-none");
    return;
  }
  if ("mediaDevices" in navigator && navigator.mediaDevices.getUserMedia) {
    const updatedConstraints = {
      ...constraints,
      video: {
        ...constraints.video,
        deviceId: {
          exact: cameraOptions.value,
        },
      },
    };
    startStream(updatedConstraints);
  }
};

const startStream = async (constraints) => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    handleStream(stream);
  } catch (error) {
    if (error.name === "OverconstrainedError") {
      console.error("Constraints could not be satisfied by available devices.", error);
      // Fallback to less strict constraints
      const fallbackConstraints = {
        video: true,
      };
      try {
        const stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
        handleStream(stream);
      } catch (fallbackError) {
        console.error("Error accessing media devices.", fallbackError);
      }
    } else {
      console.error("Error accessing media devices.", error);
    }
  }
};

const handleStream = (stream) => {
  video.srcObject = stream;
  play.classList.add("d-none");
  pause.classList.remove("d-none");
  screenshot.classList.remove("d-none");

  streamStarted = true;

  // Set up function so it could be used with interval
  function getPrediction(stream) {
    callPredictImage(stream).then((result) => {
      console.log(result);
      if (result.confidence != null && result.confidence > minConfidence * 100) {
        // Display the prediction results
        labelOutput.innerHTML = "Output: " + result.label;
        confidenceOutput.innerHTML = "Confidence: " + result.confidence + "%";
        inferenceTimeOutput.innerHTML ="Inference time: " +
          result.inferenceTime["Neural network"] + "s " +
          result.inferenceTime["Mediapipe"] + "s";
        noneCount = 0;
      }
      else {
        noneCount++;
        if (noneCount >= maxNone) {
          labelOutput.innerHTML = "Output: -";
          confidenceOutput.innerHTML = "Confidence: 0%";
          inferenceTimeOutput.innerHTML = "Inference time: 0s";
        }
      }
    });
  }

  setInterval(() => getPrediction(stream), 1000 / fps); // Interval that will predict the label of the current frame
};

getCameraSelection();
