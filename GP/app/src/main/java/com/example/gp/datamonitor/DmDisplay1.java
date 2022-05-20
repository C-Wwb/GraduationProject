package com.example.gp.datamonitor;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.gp.R;
import com.example.gp.datatransfer.DtDisplay;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

public class DmDisplay1 extends AppCompatActivity implements View.OnClickListener{

    Button bt_dmd;
    private EditText edtrmssd;
    private Button bt_rmssd;

    @Override
    protected void onCreate(Bundle saveInstanceState)
    {
        super.onCreate(saveInstanceState);
        setContentView(R.layout.dmdisplay);

        bt_dmd = findViewById(R.id.button_dmd);//点击进行监测数据传输
        bt_dmd.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(DmDisplay1.this, DtDisplay.class);
                startActivity(intent);
            }
        });

        edtrmssd = (EditText)findViewById(R.id.edt_rmssd);//点击进行rmssd数据传输
        bt_rmssd = (Button)findViewById(R.id.bt_rmssd);

        bt_rmssd.setOnClickListener(this);
    }
    public void onClick(View view)
    {
        if (view.getId() == R.id.bt_rmssd) {
            writeRmssd(edtrmssd.getText().toString());
        }
    }

    public void writeRmssd(String data)
    {
        FileOutputStream out = null;
        try {
            out = openFileOutput("client1", MODE_PRIVATE);
            out.write(data.getBytes());
            System.out.println("文件已经成功写入了");
            out.flush();
            out.close();
        }catch (FileNotFoundException e){
            e.printStackTrace();
        }catch (IOException e){
            e.printStackTrace();
        }
    }
}