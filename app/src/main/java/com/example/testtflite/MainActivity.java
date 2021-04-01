package com.example.testtflite;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.ExifInterface;
import android.os.Bundle;
import android.view.OrientationEventListener;
import android.view.Surface;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class MainActivity extends AppCompatActivity {
    private ImageAnalysis imageAnalysis;
    private OrientationEventListener orientationEventListener;
    private PoseNet net;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 123);

        }
        try {
            MappedByteBuffer kp = FileUtil.loadMappedFile(this, "posenet.tflite");
            MappedByteBuffer cl = FileUtil.loadMappedFile(this, "classify.tflite");
            net = new PoseNet(kp, cl);
            ImageView imageView = findViewById(R.id.pose);
            TextView result = findViewById(R.id.result);
            PreviewView pImg = findViewById(R.id.pImg);
            Preview preview = new Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3).build();
            imageAnalysis = new ImageAnalysis.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build();
            ExecutorService executor = Executors.newFixedThreadPool(5);
            imageAnalysis.setAnalyzer(executor, image -> {
                net.inference(image, imageView, result);
            });
            orientationEventListener = new OrientationEventListener(this) {
                @Override
                public void onOrientationChanged(int orientation) {
                    if (orientation == ExifInterface.ORIENTATION_UNDEFINED) {
                        return;
                    }
                    int rotation;
                    if (45 <= orientation && orientation <= 135) {
                        rotation = Surface.ROTATION_270;
                    } else if (135 <= orientation && orientation <= 225) {
                        rotation = Surface.ROTATION_180;
                    } else if (225 <= orientation && orientation <= 315) {
                        rotation = Surface.ROTATION_90;
                    } else {
                        rotation = Surface.ROTATION_0;
                    }
                    imageAnalysis.setTargetRotation(rotation);
                }
            };
            ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                    ProcessCameraProvider.getInstance(this);
            cameraProviderFuture.addListener(() -> {
                try {
                    ProcessCameraProvider cameraProvider = (ProcessCameraProvider) cameraProviderFuture.get();
                    CameraSelector cameraSelector = new CameraSelector.Builder()
                            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                            .build();
                    cameraProvider.bindToLifecycle(
                            this,
                            cameraSelector,
                            imageAnalysis,
                            preview);
                    preview.setSurfaceProvider(
                            pImg.getSurfaceProvider());
                } catch (InterruptedException | ExecutionException e) {

                }
            }, ContextCompat.getMainExecutor(this));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        orientationEventListener.enable();
    }

    @Override
    protected void onStop() {
        super.onStop();
        orientationEventListener.disable();
    }
}