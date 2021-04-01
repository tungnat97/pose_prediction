package com.example.testtflite;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.camera.core.ImageProxy;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CenterNet {
    private final float P_SCORE = 0.5f;
    private final float KP_SCORE = 0.5f;
    private final String[] poses = {"Hunched", "Left", "Right", "Good"};
    private final List<Keypoint> pairs;
    private final TensorImage classifyInput;
    private TensorImage keypointsInput;
    private final Interpreter keypoints;
    private final Interpreter classify;
    private final ImageProcessor kProcess;
    private final Map<Integer, Object> keypointOutputs;
    private final Map<Integer, Object> classifyOutputs;
    private final Paint paint;

    public CenterNet(MappedByteBuffer kp, MappedByteBuffer cl) {
        kProcess = new ImageProcessor.Builder()
                .add(new ResizeOp(320, 320, ResizeOp.ResizeMethod.BILINEAR))
                .build();
        paint = new Paint();
        paint.setARGB(255, 0, 255, 0);
        keypointsInput = new TensorImage(DataType.FLOAT32);
        classifyInput = new TensorImage(DataType.FLOAT32);
        keypointOutputs = new HashMap<>();
        classifyOutputs = new HashMap<>();
        pairs = new ArrayList<Keypoint>();
        pairs.add(new Keypoint(1, 3));
        pairs.add(new Keypoint(2, 4));
        pairs.add(new Keypoint(1, 0));
        pairs.add(new Keypoint(2, 0));
        pairs.add(new Keypoint(5, 6));
        pairs.add(new Keypoint(5, 7));
        pairs.add(new Keypoint(7, 9));
        pairs.add(new Keypoint(6, 8));
        pairs.add(new Keypoint(8, 10));
        pairs.add(new Keypoint(5, 11));
        pairs.add(new Keypoint(6, 12));
        pairs.add(new Keypoint(11, 12));
        pairs.add(new Keypoint(11, 13));
        pairs.add(new Keypoint(13, 15));
        pairs.add(new Keypoint(12, 14));
        pairs.add(new Keypoint(14, 16));
        CompatibilityList compatList = new CompatibilityList();
        GpuDelegate.Options gpuOptions = new GpuDelegate.Options();
        gpuOptions.setPrecisionLossAllowed(true);
        GpuDelegate gpuDelegate = new GpuDelegate(gpuOptions);
        Interpreter.Options interpreterOptions = new Interpreter.Options();
        if (compatList.isDelegateSupportedOnThisDevice()) {
            interpreterOptions.setNumThreads(4);
        } else {
            interpreterOptions.setNumThreads(4);
        }
        keypoints = new Interpreter(kp, interpreterOptions);
        classify = new Interpreter(cl, interpreterOptions);
        float[][][] out1 = new float[1][10][4];
        float[][] out2 = new float[1][10];
        float[][] out3 = new float[1][10];
        float[] out4 = new float[1];
        float[][][][] out5 = new float[1][10][17][2];
        float[][][] out6 = new float[1][10][17];
        keypointOutputs.put(0, out1);
        keypointOutputs.put(1, out2);
        keypointOutputs.put(2, out3);
        keypointOutputs.put(3, out4);
        keypointOutputs.put(4, out5);
        keypointOutputs.put(5, out6);
        float[][] out7 = new float[1][4];
        classifyOutputs.put(0, out7);
    }


    public void loadKeypointsInput(ImageProxy image, int rotation, ImageView pImg) {
        //Rotation
        byte[] nv21 = ImageUtils.YUV420toNV21(image);
        byte[] jpeg = ImageUtils.NV21toJPEG(nv21, image.getWidth(), image.getHeight(), 100);
        Bitmap inputs = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length);
        Matrix matrix = new Matrix();
        matrix.postRotate((float) rotation);
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(inputs, inputs.getWidth(), inputs.getHeight(), true);
        Bitmap rotatedBitmap = Bitmap.createBitmap(scaledBitmap, 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight(), matrix, true);
        pImg.setImageBitmap(rotatedBitmap);
        keypointsInput.load(rotatedBitmap);
        keypointsInput = kProcess.process(keypointsInput);
    }

    private boolean loadClassifyInput(ImageView imageView) {
        boolean check = false;
        Object[] inputs = {keypointsInput.getTensorBuffer().getBuffer()};
        keypoints.runForMultipleInputsOutputs(inputs, keypointOutputs);
        Bitmap black_image = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888);

        Canvas canvas = new Canvas(black_image);
        canvas.drawColor(Color.BLACK);

        float max_x = -1000f;
        float min_x = 1000f;
        float max_y = -1000f;
        float min_y = 1000f;
        float[][][] probArray = (float[][][]) keypointOutputs.get(5);
        float[][][][] coor = (float[][][][]) (keypointOutputs.get(4));
        float[][] objectProb = (float[][]) (keypointOutputs.get(2));
        for (int i = 0; i < 17; ++i) {
            float p = probArray[0][0][i];
            if (p > KP_SCORE) {
                max_x = Math.max(max_x, coor[0][0][i][0]);
                min_x = Math.min(min_x, coor[0][0][i][0]);
                max_y = Math.max(max_y, coor[0][0][i][1]);
                min_y = Math.min(min_y, coor[0][0][i][1]);
            }
        }
        if (max_x == -1000f) {
            max_x = 1;
            min_x = 0;
        }
        if (max_y == -1000f) {
            max_y = 1;
            min_y = 0;
        }
        float pScore = objectProb[0][0];
        if (pScore > P_SCORE) {
            check = true;
            for (Keypoint pair : pairs) {
                float p0 = probArray[0][0][pair.x];
                float p1 = probArray[0][0][pair.y];
                if (p0 > KP_SCORE && p1 > KP_SCORE) {
                    float x0 = (coor[0][0][pair.x][0] - min_x) / (max_x - min_x) * 128;
                    float y0 = (coor[0][0][pair.x][1] - min_y) / (max_y - min_y) * 128;
                    float x1 = (coor[0][0][pair.y][0] - min_x) / (max_x - min_x) * 128;
                    float y1 = (coor[0][0][pair.y][1] - min_y) / (max_y - min_y) * 128;
                    canvas.drawLine(y0, x0, y1, x1, paint);
                }
            }
        }

        imageView.setImageBitmap(black_image);
        if (check) {
            classifyInput.load(black_image);
        }
        return check;
    }

    public void inference(ImageProxy image, ImageView pImg, ImageView imageView, TextView result) {
        try {
            int rotation = image.getImageInfo().getRotationDegrees();
            if (image != null) {
                loadKeypointsInput(image, rotation, pImg);
                boolean check = loadClassifyInput(imageView);
                if (check) {
                    Object[] inputs = {classifyInput.getTensorBuffer().getBuffer()};
                    classify.runForMultipleInputsOutputs(inputs, classifyOutputs);
                    float[][] prob = (float[][]) classifyOutputs.get(0);
                    result.setText(poses[getIndexOfLargest(prob[0])]);
                } else {
                    result.setText("No person found");
                }
            }
        } catch (Exception e) {
            Log.e("ANALYZE", e.getMessage(), e);
        } finally {
            image.close();
        }
    }

    public int getIndexOfLargest(float[] array) {
        if (array == null || array.length == 0) {
            return -1;
        }
        int largest = 0;
        for (int i = 1; i < array.length; ++i) {
            if (array[i] > array[largest]) {
                largest = i;
            }
        }
        return largest;
    }

    protected static class Keypoint {
        protected int x;
        protected int y;

        protected Keypoint(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }
}
