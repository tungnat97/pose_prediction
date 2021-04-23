package com.ais.dangngoi;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import com.ais.dangngoi.screens.CameraView;
import com.ais.dangngoi.screens.PickImage;
import com.ais.dangngoi.screens.SensorLight;

import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.MappedByteBuffer;


public class MainActivity extends AppCompatActivity {
    /**
     * Khai báo content trong view
     */
    private Button buttonVideo, buttonImage, buttonBrightness;
    /**
     * Constant
     */
    private String TAG = "Mainactivity";
    private static final String[] CAMERA_PERMISSION = new String[]{Manifest.permission.CAMERA};
    private static final int CAMERA_REQUEST_CODE = 10;

//    private ImageView imageViewLogo;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initializationView();
        handleOnClickButton();
    }

    private void handleOnClickButton() {
        buttonVideo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(TAG, "On button video");
                if (hasCameraPermission()) {
                    Intent intent1 = new Intent(MainActivity.this, CameraView.class);
                    startActivity(intent1);
                } else {
                    requestPermission();
                }
            }
        });
        buttonImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(TAG, "On button image");
                if (hasCameraPermission()) {
                    Intent intent1 = new Intent(MainActivity.this, PickImage.class);
                    startActivity(intent1);
                } else {
                    requestPermission();
                }
            }
        });
        buttonBrightness.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(TAG, "On button brightness");
                Intent intent1 = new Intent(MainActivity.this, SensorLight.class);
                startActivity(intent1);
            }
        });
    }

    private void initializationView() {
        buttonVideo = findViewById(R.id.buttonVideo);
        buttonImage = findViewById(R.id.buttonImage);
        buttonBrightness = findViewById(R.id.buttonLight);
    }

    private boolean hasCameraPermission() {
        return ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED;
    }


    private void requestPermission() {
        ActivityCompat.requestPermissions(
                this,
                CAMERA_PERMISSION,
                CAMERA_REQUEST_CODE
        );
    }
}