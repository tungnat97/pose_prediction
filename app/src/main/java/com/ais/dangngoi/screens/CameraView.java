package com.ais.dangngoi.screens;

import android.annotation.SuppressLint;
import android.media.ExifInterface;
import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;
import android.view.OrientationEventListener;
import android.view.Surface;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.core.UseCaseGroup;
import androidx.camera.core.ViewPort;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.ais.dangngoi.PoseNet;
import com.ais.dangngoi.R;
import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CameraView extends AppCompatActivity {
    /**
     * Constant
     */
    private final Float[] listInterval = new Float[]{5f, 10f, 15f};
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private TextView textView;
    private PreviewView previewView;
    private Spinner spinnerInterval;
    private PoseNet net;
    private float time;
    private float good;
    private float total;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera_view);
        previewView = findViewById(R.id.previewView);
        getSupportActionBar().setDisplayHomeAsUpEnabled(true); // Enable button back in header
        initializationView();
        try {
            MappedByteBuffer kp = FileUtil.loadMappedFile(this, "posenet.tflite");
            MappedByteBuffer cl = FileUtil.loadMappedFile(this, "classify.tflite");
            net = new PoseNet(kp, cl);
        } catch (IOException e) {
            e.printStackTrace();
        }
        time = 0f;
        good = 0f;
        total = 0f;
        /**
         * Connect preview view camera
         */
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        textView = findViewById(R.id.camera_processing);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindImageAnalysis(cameraProvider);
            } catch (ExecutionException | InterruptedException ignored) {
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void initializationView() {
        spinnerInterval = findViewById(R.id.spinner_interval);
        ArrayAdapter<Float> adapter = new ArrayAdapter<Float>(this, android.R.layout.simple_spinner_dropdown_item, listInterval);
        spinnerInterval.setAdapter(adapter);
    }


    /**
     * Fill data from camera to preview
     *
     * @param cameraProvider
     */
    @SuppressLint("UnsafeExperimentalUsageError")
    private void bindImageAnalysis(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();
        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build();
        ExecutorService executor = Executors.newFixedThreadPool(4);
        imageAnalysis.setAnalyzer(executor, image -> {
            long start = System.currentTimeMillis();
            float res = net.inference(image);
            if (res != -1) {
                if (res > 0.5) {
                    good += 1f;
                }
                total += 1f;
            }
            long end = System.currentTimeMillis();
            time += (float) (end - start) / 1000;
            if (time >= listInterval[spinnerInterval.getSelectedItemPosition()]) {
                float percent = good / Math.max(total, 1f) * 100;
                float interval = listInterval[spinnerInterval.getSelectedItemPosition()];
                String text = String.format("Trong %.2f giây, ngồi tốt %.2f phần trăm.", interval, percent);
                textView.setText(text);
                time = 0;
                good = 0;
                total = 0;
            }
            image.close();
        });
        OrientationEventListener orientationEventListener = new OrientationEventListener(this) {
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
        orientationEventListener.enable();
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK).build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        ViewPort viewPort = previewView.getViewPort();
        UseCaseGroup useCaseGroup = new UseCaseGroup.Builder()
                .setViewPort(viewPort)
                .addUseCase(preview)
                .addUseCase(imageAnalysis)
                .build();
        cameraProvider.bindToLifecycle(this, cameraSelector,
                useCaseGroup);
    }

    /**
     * Back button listener
     */
    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                super.finish();
                return true;
        }

        return super.onOptionsItemSelected(item);
    }
}