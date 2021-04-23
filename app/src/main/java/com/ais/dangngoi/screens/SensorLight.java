package com.ais.dangngoi.screens;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.view.MenuItem;
import android.widget.TextView;

import com.ais.dangngoi.R;

public class SensorLight extends AppCompatActivity {
    TextView textLIGHT_available, textLIGHT_reading;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sensor_light);
        getSupportActionBar().setDisplayHomeAsUpEnabled(true); // Enable button back in header

        textLIGHT_available
                = (TextView) findViewById(R.id.light_sensor_available);
        textLIGHT_reading
                = (TextView) findViewById(R.id.sensor_reading_data);

        SensorManager mySensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);

        Sensor lightSensor = mySensorManager.getDefaultSensor(Sensor.TYPE_LIGHT);
        if (lightSensor != null) {
            mySensorManager.registerListener(
                    lightSensorListener,
                    lightSensor,
                    SensorManager.SENSOR_DELAY_NORMAL);

        } else {
            textLIGHT_available.setText("Cảm biến ánh sáng không khả dụng ");
        }
    }

    private final SensorEventListener lightSensorListener
            = new SensorEventListener() {

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {
            // TODO Auto-generated method stub

        }

        @Override
        public void onSensorChanged(SensorEvent event) {
            String res = "";
            if (event.sensor.getType() == Sensor.TYPE_LIGHT) {
                if (event.values[0] >= 300 && event.values[0] <= 500) {
                    res = "\nÁnh sáng ổn.";
                } else if (event.values[0] < 300) {
                    res = "\nQuá tối";
                } else {
                    res = "\nQuá sáng";
                }
                textLIGHT_reading.setText("LUX level: " + event.values[0] + res);
            }
        }

    };

    /**
     * Back button listener
     */
    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        if (item.getItemId() == android.R.id.home) {
            super.finish();
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
}