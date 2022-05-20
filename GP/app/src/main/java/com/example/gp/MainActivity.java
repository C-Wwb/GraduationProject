package com.example.gp;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import com.example.gp.datamonitor.DmDisplay;
import com.example.gp.datamonitor.DmDisplay1;

public class MainActivity extends AppCompatActivity {
    EditText name;
    EditText passwd;
    Button bt_log;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        name = findViewById(R.id.log_name);
        passwd = findViewById(R.id.log_password);

        bt_log = findViewById(R.id.button_log);
        bt_log.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MainActivity.this, DmDisplay.class);
                startActivity(intent);
            }
        });

    }

    public void Login(View v)
    {
        String UserName = "wwb";
        String UserPassword = "123456";

        String user = name.getText().toString().trim();
        String pwd = passwd.getText().toString().trim();

        if(user.equals(UserName) & pwd.equals(UserPassword))
        {
            Toast.makeText(this, "成功", Toast.LENGTH_SHORT).show();
        }else{
            Toast.makeText(this, "身份验证错误，禁止访问", Toast.LENGTH_SHORT).show();
        }
    }
}