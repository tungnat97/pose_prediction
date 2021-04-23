package com.ais.dangngoi;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.media.ExifInterface;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.camera.core.ImageProxy;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.IOException;
import java.io.Serializable;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static android.graphics.Bitmap.createBitmap;
import static android.graphics.Bitmap.createScaledBitmap;
import static com.ais.dangngoi.ImageUtils.NV21toJPEG;
import static com.ais.dangngoi.ImageUtils.YUV420toNV21;

public class PoseNet {
    private final float P_SCORE = 0.2f;
    private final float KP_SCORE = 0.5f;
    private final String[] poses = {"Bad", "Good"};
    private final List<KeyPoint> pairs;
    private final TensorImage classifyInput;
    private final Interpreter keyPoints;
    private final Interpreter classify;
    private final ImageProcessor kProcess;
    private final Map<Integer, Object> keyPointsOutput;
    private final Map<Integer, Object> classifyOutputs;
    private final Paint paint;
    private TensorImage keyPointsInput;

    public PoseNet(MappedByteBuffer kp, MappedByteBuffer cl) {
        kProcess = new ImageProcessor.Builder()
                .add(new ResizeOp(257, 257, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(128.0f, 128.0f))
                .build();
        paint = new Paint();
        paint.setARGB(255, 0, 255, 0);
        keyPointsInput = new TensorImage(DataType.FLOAT32);
        classifyInput = new TensorImage(DataType.FLOAT32);
        keyPointsOutput = new HashMap<>();
        classifyOutputs = new HashMap<>();
        pairs = new ArrayList<>();
        pairs.add(new KeyPoint(1, 3));
        pairs.add(new KeyPoint(2, 4));
        pairs.add(new KeyPoint(1, 0));
        pairs.add(new KeyPoint(2, 0));
        pairs.add(new KeyPoint(5, 6));
        pairs.add(new KeyPoint(5, 7));
        pairs.add(new KeyPoint(7, 9));
        pairs.add(new KeyPoint(6, 8));
        pairs.add(new KeyPoint(8, 10));
        pairs.add(new KeyPoint(5, 11));
        pairs.add(new KeyPoint(6, 12));
        pairs.add(new KeyPoint(11, 12));
        pairs.add(new KeyPoint(11, 13));
        pairs.add(new KeyPoint(13, 15));
        pairs.add(new KeyPoint(12, 14));
        pairs.add(new KeyPoint(14, 16));
        CompatibilityList compatList = new CompatibilityList();
        GpuDelegate.Options gpuOptions = new GpuDelegate.Options();
        gpuOptions.setPrecisionLossAllowed(true);
        GpuDelegate gpuDelegate = new GpuDelegate(gpuOptions);
        Interpreter.Options interpreterOptions = new Interpreter.Options();
        if (compatList.isDelegateSupportedOnThisDevice()) {
            interpreterOptions.addDelegate(gpuDelegate);
        } else {
            interpreterOptions.setNumThreads(4);
        }
        keyPoints = new Interpreter(kp, interpreterOptions);
        classify = new Interpreter(cl, interpreterOptions);
        float[][][][] out1 = new float[1][9][9][17];
        float[][][][] out2 = new float[1][9][9][34];
        float[][][][] out3 = new float[1][9][9][32];
        float[][][][] out4 = new float[1][9][9][32];
        keyPointsOutput.put(0, out1);
        keyPointsOutput.put(1, out2);
        keyPointsOutput.put(2, out3);
        keyPointsOutput.put(3, out4);
        float[][] out5 = new float[1][1];
        classifyOutputs.put(0, out5);
    }

    public void loadKeyPointsBitmap(String path, Bitmap bitmap, ImageView imageView) throws IOException {
        ExifInterface exif = new ExifInterface(path);
        int rotation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
        int rotationInDegrees = exifToDegrees(rotation);
        Matrix matrix = new Matrix();
        matrix.postRotate((float) rotationInDegrees);
        Bitmap scaledBitmap = createScaledBitmap(bitmap,
                bitmap.getWidth(),
                bitmap.getHeight(),
                true);
        Bitmap finalInputs = createBitmap(scaledBitmap,
                0,
                0,
                scaledBitmap.getWidth(),
                scaledBitmap.getHeight(),
                matrix,
                true);
        imageView.setImageBitmap(finalInputs);
        keyPointsInput.load(finalInputs);
        keyPointsInput = kProcess.process(keyPointsInput);
    }

    public void loadKeyPointsInput(ImageProxy image, int rotation) {
        byte[] jpeg = NV21toJPEG(YUV420toNV21(image),
                image.getWidth(),
                image.getHeight(),
                100);
        //Rotation
        Rect crop = image.getCropRect();
        Bitmap inputs = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length);
        inputs = Bitmap.createBitmap(inputs, crop.left, crop.top, crop.right - crop.left, crop.bottom - crop.top);
        Matrix matrix = new Matrix();
        matrix.postRotate((float) rotation);
        Bitmap scaledBitmap = createScaledBitmap(inputs,
                inputs.getWidth(),
                inputs.getHeight(),
                true);
        Bitmap finalInputs = createBitmap(scaledBitmap,
                0,
                0,
                scaledBitmap.getWidth(),
                scaledBitmap.getHeight(),
                matrix,
                true);
        keyPointsInput.load(finalInputs);
        keyPointsInput = kProcess.process(keyPointsInput);
    }

    private boolean loadClassifyInput() {
        boolean check = false;
        Object[] inputs = {keyPointsInput.getTensorBuffer().getBuffer()};
        keyPoints.runForMultipleInputsOutputs(inputs, keyPointsOutput);

        Bitmap black_image = createBitmap(128, 128, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(black_image);
        canvas.drawColor(Color.BLACK);
        float[][][][] heatMap = (float[][][][]) keyPointsOutput.get(0);
        float[][][][] offset = (float[][][][]) (keyPointsOutput.get(1));

        int height = heatMap[0].length;
        int width = heatMap[0][0].length;
        int numKeyPoints = heatMap[0][0][0].length;

        KeyPoint[] keyPointPositions = new KeyPoint[numKeyPoints];
        for (int keypoint = 0; keypoint < numKeyPoints; ++keypoint) {
            float maxVal = heatMap[0][0][0][keypoint];
            int maxRow = 0;
            int maxCol = 0;
            for (int row = 0; row < height; ++row) {
                for (int col = 0; col < width; ++col) {
                    if (heatMap[0][row][col][keypoint] > maxVal) {
                        maxVal = heatMap[0][row][col][keypoint];
                        maxRow = row;
                        maxCol = col;
                    }
                }
            }
            keyPointPositions[keypoint] = new KeyPoint(maxRow, maxCol);
        }
        int[] xCoords = new int[numKeyPoints];
        int[] yCoords = new int[numKeyPoints];
        float[] confidenceScores = new float[numKeyPoints];
        for (int idx = 0; idx < numKeyPoints; ++idx) {
            int positionY = keyPointPositions[idx].x;
            int positionX = keyPointPositions[idx].y;
            yCoords[idx] = (int) (positionY / (float) (height - 1) * 257f +
                    offset[0][positionY][positionX][idx]);
            xCoords[idx] = (int) (positionX / (float) (width - 1) * 257f +
                    offset[0][positionY][positionX][idx + numKeyPoints]);
            confidenceScores[idx] = sigmoid(heatMap[0][positionY][positionX][idx]);
        }
        float pScore = 0;
        for (float x : confidenceScores) {
            pScore += x;
        }
        pScore /= confidenceScores.length;
        int max_x = -1000;
        int min_x = 1000;
        int max_y = -1000;
        int min_y = 1000;
        for (int i = 0; i < 17; ++i) {
            if (confidenceScores[i] > KP_SCORE) {
                max_x = Math.max(max_x, xCoords[i]);
                min_x = Math.min(min_x, xCoords[i]);
                max_y = Math.max(max_y, yCoords[i]);
                min_y = Math.min(min_y, yCoords[i]);
            }
        }
        if (pScore > P_SCORE) {
            check = true;
            for (KeyPoint pair : pairs) {
                if (confidenceScores[pair.x] > KP_SCORE && confidenceScores[pair.y] > KP_SCORE) {
                    float x0 = (float) (xCoords[pair.x] - min_x) / (max_x - min_x) * 128;
                    float y0 = (float) (yCoords[pair.x] - min_y) / (max_y - min_y) * 128;
                    float x1 = (float) (xCoords[pair.y] - min_x) / (max_x - min_x) * 128;
                    float y1 = (float) (yCoords[pair.y] - min_y) / (max_y - min_y) * 128;
                    canvas.drawLine(x0, y0, x1, y1, paint);
                }
            }
        }
        if (check) {
            classifyInput.load(black_image);
        }
        return check;
    }

    @SuppressLint({"SetTextI18n", "DefaultLocale"})
    public float inference(ImageProxy image) {
        float res = -1f;
        try {
            int rotation = image.getImageInfo().getRotationDegrees();
            if (image != null) {
                loadKeyPointsInput(image, rotation);
                boolean check = loadClassifyInput();
                if (check) {
                    Object[] inputs = {classifyInput.getTensorBuffer().getBuffer()};
                    classify.runForMultipleInputsOutputs(inputs, classifyOutputs);
                    float[][] prob = (float[][]) classifyOutputs.get(0);
                    res = prob[0][0];
                }
            }
        } catch (Exception e) {
            Log.w("ANALYZE", e.getMessage());
        }
        return res;
    }

    public float inferenceOneShot(String path, Bitmap bitmap, ImageView imageView) {
        float res = -1f;
        try {
            loadKeyPointsBitmap(path, bitmap, imageView);
            boolean check = loadClassifyInput();
            if (check) {
                Object[] inputs = {classifyInput.getTensorBuffer().getBuffer()};
                classify.runForMultipleInputsOutputs(inputs, classifyOutputs);
                float[][] prob = (float[][]) classifyOutputs.get(0);
                res = prob[0][0];
            }
        } catch (IOException e) {
            Log.w("ANALYZE", e.getMessage());
        }
        return res;
    }
//    public int getIndexOfLargest(float[] array) {
//        if (array == null || array.length == 0) {
//            return -1;
//        }
//        int largest = 0;
//        for (int i = 1; i < array.length; ++i) {
//            if (array[i] > array[largest]) {
//                largest = i;
//            }
//        }
//        return largest;
//    }

    private float sigmoid(Float x) {
        return (float) (1.0f / (1.0f + Math.exp(-x)));
    }

    protected static class KeyPoint {
        protected int x;
        protected int y;

        protected KeyPoint(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    private static int exifToDegrees(int exifOrientation) {
        if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_90) { return 90; }
        else if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_180) {  return 180; }
        else if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_270) {  return 270; }
        return 0;
    }
}