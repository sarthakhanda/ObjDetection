package org.tensorflow.lite.examples.detection.tflite;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Trace;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;
import org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions;

public class TFLiteObjectDetectionAPIModel implements Detector {
  private static final String TAG = "TFLiteObjectDetectionAPIModelWithTaskApi";

  /** Only return this many results. */
  private static final int NUM_DETECTIONS = 10;

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  private final ObjectDetector objectDetector;

  public static Detector create(
      final Context context,
      final String modelFilename,
      final String labelFilename,
      final int inputSize,
      final boolean isQuantized)
      throws IOException {
    return new TFLiteObjectDetectionAPIModel(context, modelFilename);
  }

  private TFLiteObjectDetectionAPIModel(Context context, String modelFilename) throws IOException {
    ObjectDetectorOptions options =
        ObjectDetectorOptions.builder().setMaxResults(NUM_DETECTIONS).build();
    objectDetector = ObjectDetector.createFromFileAndOptions(context, modelFilename, options);
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");
    List<Detection> results = objectDetector.detect(TensorImage.fromBitmap(bitmap));

    // Converts a list of {@link Detection} objects into a list of {@link Recognition} objects
    // to match the interface of other inference method, such as using the <a
    // href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android/lib_interpreter">TFLite
    // Java API.</a>.
    final ArrayList<Recognition> recognitions = new ArrayList<>();
    int cnt = 0;
    for (Detection detection : results) {
      recognitions.add(
          new Recognition(
              "" + cnt++,
              detection.getCategories().get(0).getLabel(),
              detection.getCategories().get(0).getScore(),
              detection.getBoundingBox()));
    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {
    if (objectDetector != null) {
      objectDetector.close();
    }
  }

  @Override
  public void setNumThreads(int numThreads) {
    if (numThreads != 1) {
      throw new IllegalArgumentException(
          "Manipulating the numbers of threads is not allowed in the Task"
              + " library currently. The current implementation runs on single thread.");
    }
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    throw new UnsupportedOperationException(
        "Manipulating the hardware accelerators is not allowed in the Task"
            + " library currently. Only CPU is allowed.");
  }
}
