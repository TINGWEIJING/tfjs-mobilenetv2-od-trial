import "./App.css";
import React, { useEffect, useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import ic_img from "./Holding IC.png";

const videoConstraints = {
    height: 1080,
    width: 1920,
    maxWidth: "100vw",
    facingMode: "environment",
};

const face_detection_url = "tfjs/face_detection/model.json";

const buildDetectedObjects = (scores, threshold, boxes, classes, height, width) => {
    const detectionObjects = [];

    scores.forEach((score, i) => {
        if (score > threshold) {
            const bbox = [];
            const minY = boxes[0][i][0] * height;
            const minX = boxes[0][i][1] * width;
            const maxY = boxes[0][i][2] * height;
            const maxX = boxes[0][i][3] * width;
            bbox[0] = minX;
            bbox[1] = minY;
            bbox[2] = maxX - minX;
            bbox[3] = maxY - minY;
            detectionObjects.push({
                class: classes[i],
                label: "face",
                score: score.toFixed(4),
                bbox: bbox,
            });
        }
    });
    return detectionObjects;
};

const readImageFile = (file) => {
    return new Promise((resolve) => {
        const reader = new FileReader();

        reader.onload = () => resolve(reader.result);

        reader.readAsDataURL(file);
    });
};

const createHTMLImageElement = (imageSrc) => {
    return new Promise((resolve) => {
        const img = new Image();

        img.onload = () => resolve(img);

        img.src = imageSrc;
    });
};

function App() {
    const webcamRef = useRef(null);
    const [model, setModel] = useState(null);
    const [videoWidth, setVideoWidth] = useState(960);
    const [videoHeight, setVideoHeight] = useState(640);

    const loadModel = async () => {
        /** @type {tf.GraphModel} */
        const loadedModel = await tf.loadGraphModel(face_detection_url);
        setModel(loadedModel);
        console.log("Model loaded.");
        return loadedModel;
    };

    const onCapture = async (model) => {
        // console.log("Capturing");
        // Check data is available
        if (typeof webcamRef.current !== "undefined" && webcamRef.current !== null && webcamRef.current.video.readyState === 4 && model !== null) {
            // if (true) {
            // Get Video Properties
            /** @type {HTMLVideoElement} */
            const video = webcamRef.current.video;
            const videoWidth = video.videoWidth;
            const videoHeight = video.videoHeight;

            // Set video width
            webcamRef.current.video.width = videoWidth;
            webcamRef.current.video.height = videoHeight;

            const cnvs = document.getElementById("myCanvas");
            cnvs.width = videoWidth;
            cnvs.height = videoHeight;
            // cnvs.style.position = "absolute";

            const ctx = cnvs.getContext("2d");

            // Font options.
            const font = "16px sans-serif";
            ctx.font = font;
            ctx.textBaseline = "top";

            // const ic_img = document.getElementById("ic_img");
            /** @type {tf.Tensor3D} */
            const rawImgTensor = tf.browser.fromPixels(video);
            // const rawImgTensor = tf.browser.fromPixels(ic_img);
            // console.log(`rawImgTensor shape: ${rawImgTensor.shape}`);

            // const [inputTensorWidth, inputTensorHeight] = model.inputs[0].shape.slice(1, 3); // [640, 640]
            // const inputTensor = tf.tidy(() => {
            //     return tf.image.resizeBilinear(rawImgTensor, [inputTensorWidth, inputTensorHeight]).div(255.0).expandDims(0);
            // });
            const inputTensor = tf.tidy(() => {
                return rawImgTensor.transpose([0, 1, 2]).expandDims();
            });
            // console.log(`inputTensor shape: ${inputTensor.shape}`);
            let startTime = performance.now();
            model
                .executeAsync(inputTensor)
                .then((res) => {
                    // const a0 = res[0].arraySync(); // num_detection
                    // const a1 = res[1].arraySync(); // raw_detection_boxes
                    // const a2 = res[2].arraySync(); // detection_anchor_indices
                    // const a3 = res[3].arraySync(); // raw_detection_scores
                    // const a4 = res[4].arraySync(); // detection_boxes
                    // const a5 = res[5].arraySync(); // detection_classes
                    // const a6 = res[6].arraySync(); // detection_scores
                    // const a7 = res[7].arraySync(); // detection_multiclass_scores
                    const detection_boxes = res[4].arraySync();
                    const detection_classes = res[5].arraySync();
                    const detection_scores = res[6].dataSync();
                    const detections = buildDetectedObjects(detection_scores, 0.5, detection_boxes, detection_classes, videoHeight, videoWidth);
                    ctx.clearRect(0, 0, webcamRef.current.video.videoWidth, webcamRef.current.video.videoHeight);
                    detections.forEach((item) => {
                        const x = item["bbox"][0];
                        const y = item["bbox"][1];
                        const width = item["bbox"][2];
                        const height = item["bbox"][3];

                        // Draw the bounding box.
                        ctx.strokeStyle = "#00FFFF";
                        ctx.lineWidth = 4;
                        ctx.strokeRect(x, y, width, height);

                        // Draw the label background.
                        ctx.fillStyle = "#00FFFF";
                        const textWidth = ctx.measureText(item["label"] + " " + (100 * item["score"]).toFixed(2) + "%").width;
                        const textHeight = parseInt(font, 10); // base 10
                        ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
                    });

                    detections.forEach((item) => {
                        const x = item["bbox"][0];
                        const y = item["bbox"][1];

                        // Draw the text last to ensure it's on top.
                        ctx.fillStyle = "#000000";
                        ctx.fillText(item["label"] + " " + (100 * item["score"]).toFixed(2) + "%", x, y);
                    });
                    // let endTime = performance.now();
                    // console.log(`Took ${endTime - startTime} milliseconds`);
                    return res;
                })
                .then((res) => {
                    let i = 0;
                    const len = res.length;
                    while (i < len) {
                        tf.dispose(res[i]);
                        i++;
                    }
                })
                .finally(() => {
                    tf.dispose(rawImgTensor);
                    tf.dispose(inputTensor);
                });
            // console.dir(`numTensors: ${tf.memory().numTensors}`);
        }
    };

    /* 
    Run only once
     */
    useEffect(() => {
        // console.log(tfgl.version_webgl);
        // console.log(tf.getBackend());
        // tfgl.webgl.forceHalfFloat();
        // var maxSize = tfgl.webgl_util.getWebGLMaxTextureSize(tfgl.version_webgl);
        // console.log(maxSize);
        tf.ready()
            .then((_) => {
                tf.enableProdMode();
                console.log("tfjs is ready");
            })
            .then(loadModel)
            .then((loadedModel) => {
                console.log("Test model is loaded");
                setInterval(onCapture, 100, loadedModel);
                // onCapture(loadedModel);
            });
    }, []);

    // let supportedConstraints = navigator.mediaDevices.getSupportedConstraints();
    // console.log(supportedConstraints);
    return (
        <div className="App">
            <div style={{ position: "absolute", top: "0px", zIndex: "9999" }}>
                <canvas id="myCanvas" width={videoWidth} height={videoHeight} style={{ backgroundColor: "transparent" }} />
            </div>
            <div style={{ position: "absolute", top: "0px" }}>
                <Webcam
                    audio={false}
                    id="img"
                    ref={webcamRef}
                    // width={640}
                    screenshotQuality={1}
                    screenshotFormat="image/jpeg"
                    videoConstraints={videoConstraints}
                />
                {/* <img id="ic_img" src={ic_img} alt="" /> */}
            </div>
        </div>
    );
}

export default App;
