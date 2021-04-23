package com.ais.dangngoi.screens;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import com.ais.dangngoi.MainActivity;
import com.bumptech.glide.Glide;

import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.ais.dangngoi.BuildConfig;
import com.ais.dangngoi.PoseNet;
import com.ais.dangngoi.R;
import com.bumptech.glide.load.DataSource;
import com.bumptech.glide.load.engine.GlideException;
import com.bumptech.glide.request.RequestListener;
import com.bumptech.glide.request.target.CustomTarget;
import com.bumptech.glide.request.target.Target;
import com.bumptech.glide.request.transition.Transition;

import org.tensorflow.lite.support.common.FileUtil;

import java.io.File;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.ExecutionException;

public class PickImage extends AppCompatActivity {
    private Button buttonGallery, buttonCapture;
    private ImageView imagePicked;
    /**
     * Result process
     */
    private TextView textViewResult;
    private File fileImage;
    private PoseNet net;

    /**
     * Constant
     */
    private final int CAPTURE = 1;
    private final int GALLERY = 2;
    private final String fileNameStore = "fileCapture.jpg";
    private final String TAG = "PickImage";
    final int MyVersion = Build.VERSION.SDK_INT;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_pick_image);
        getSupportActionBar().setDisplayHomeAsUpEnabled(true); // Enable button back in header

        /**
         * Initialization view
         */
//        buttonGallery = findViewById(R.id.button_pick_image);
        buttonCapture = findViewById(R.id.button_capture_image);
        imagePicked = findViewById(R.id.pick_imageview);
        textViewResult = findViewById(R.id.pick_processing_text);
        try {
            MappedByteBuffer kp = FileUtil.loadMappedFile(this, "posenet.tflite");
            MappedByteBuffer cl = FileUtil.loadMappedFile(this, "classify.tflite");
            net = new PoseNet(kp, cl);
        } catch (IOException e) {
            e.printStackTrace();
        }
        buttonListener();
    }


    private void buttonListener() {
//        buttonGallery.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                Log.d(TAG, "onClick: choose image from gallery");
//                Intent pickPhoto = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
//                startActivityForResult(pickPhoto, GALLERY);
//            }
//
//        });
        buttonCapture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(TAG, "onClick:  capture new image");
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                fileImage = getImageUri();
                Uri fileProvider = FileProvider.getUriForFile(PickImage.this, BuildConfig.APPLICATION_ID + ".fileprovider", fileImage);
                intent.putExtra(MediaStore.EXTRA_OUTPUT, fileProvider);
                startActivityForResult(intent, CAPTURE);
            }

        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (requestCode == CAPTURE) {
                Bitmap takenImage = BitmapFactory.decodeFile(fileImage.getAbsolutePath());
                float res = net.inferenceOneShot(fileImage.getAbsolutePath(), takenImage, imagePicked);
                if (res != -1) {
                    if (res > 0.5) {
                        textViewResult.setText(String.format("Dáng tốt, điểm: %.2f", res));
                    } else {
                        textViewResult.setText(String.format("Dáng xấu, điểm: %.2f", res));
                    }
                } else {
                    textViewResult.setText("Không có người trong ảnh");
                }
            }
//            else if (requestCode == GALLERY) {
//                Uri selectedImage = data.getData();
//                String[] filePath = {MediaStore.Images.Media.DATA};
//                Cursor c = getContentResolver().query(selectedImage, filePath, null, null, null);
//                c.moveToFirst();
//                int columnIndex = c.getColumnIndex(filePath[0]);
//                String picturePath = c.getString(columnIndex);
//                c.close();
//                Bitmap thumbnail = BitmapFactory.decodeFile(picturePath);
//                imagePicked.setImageBitmap(thumbnail);
//            }
        }
    }

    private File getImageUri() {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        return new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "share_image_" + System.currentTimeMillis() + ".png");

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