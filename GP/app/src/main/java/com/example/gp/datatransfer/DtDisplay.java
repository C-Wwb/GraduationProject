package com.example.gp.datatransfer;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;
import com.example.gp.R;

public class DtDisplay extends  AppCompatActivity{

    Button bt_rd;
    @Override
    protected void onCreate(Bundle saveInstanceState) {
        super.onCreate(saveInstanceState);
        setContentView(R.layout.dtdisplay);

        bt_rd = findViewById(R.id.button_rd);
        final Uri uri = Uri.parse("https://baidu.com");
        bt_rd.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_VIEW, uri);
                startActivity(intent);
            }
        });
    }
}