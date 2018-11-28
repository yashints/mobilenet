import { Component, ViewChild, ElementRef, OnInit } from "@angular/core";
import * as tf from "@tensorflow/tfjs";
import { IMAGENET_CLASSES } from "../assets/imagenet-classes";

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 5;
@Component({
  selector: "app-root",
  templateUrl: "./app.component.html",
  styleUrls: ["./app.component.scss"]
})
export class AppComponent implements OnInit {
  model: tf.Model;
  classes: any[];
  imageData: ImageData;

  @ViewChild("chosenImage") img: ElementRef;
  @ViewChild("fileUpload") fileUpload: ElementRef;

  ngOnInit() {
    this.loadModel();
  }

  async loadModel() {
    this.model = await tf.loadModel("../assets/model.json");
  }

  fileChangeEvent(event: any) {
    const file = event.target.files[0];
    if (!file || !file.type.match("image.*")) {
      return;
    }

    this.classes = [];

    const reader = new FileReader();
    reader.onload = e => {
      this.img.nativeElement.src = e.target["result"];
      this.predict(this.img.nativeElement);
    };
    reader.readAsDataURL(file);
  }

  async predict(imageData: ImageData): Promise<any> {
    this.fileUpload.nativeElement.value = "";
    const startTime = performance.now();
    const logits = tf.tidy(() => {
      // tf.fromPixels() returns a Tensor from an image element.
      const img = tf.fromPixels(imageData).toFloat();

      const offset = tf.scalar(127.5);
      // Normalize the image from [0, 255] to [-1, 1].
      const normalized = img.sub(offset).div(offset);

      // Reshape to a single-element batch so we can pass it to predict.
      const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

      // Make a prediction through mobilenet.
      return this.model.predict(batched);
    });

    // Convert logits to probabilities and class names.
    this.classes = await this.getTopKClasses(logits, TOPK_PREDICTIONS);
    const totalTime = performance.now() - startTime;
    console.log(`Done in ${Math.floor(totalTime)}ms`);
  }

  async getTopKClasses(logits, topK): Promise<any[]> {
    const values = await logits.data();

    const valuesAndIndices = [];
    for (let i = 0; i < values.length; i++) {
      valuesAndIndices.push({ value: values[i], index: i });
    }
    valuesAndIndices.sort((a, b) => {
      return b.value - a.value;
    });
    const topkValues = new Float32Array(topK);
    const topkIndices = new Int32Array(topK);
    for (let i = 0; i < topK; i++) {
      topkValues[i] = valuesAndIndices[i].value;
      topkIndices[i] = valuesAndIndices[i].index;
    }

    const topClassesAndProbs = [];
    for (let i = 0; i < topkIndices.length; i++) {
      topClassesAndProbs.push({
        className: IMAGENET_CLASSES[topkIndices[i]],
        probability: topkValues[i]
      });
    }
    return topClassesAndProbs;
  }
}
