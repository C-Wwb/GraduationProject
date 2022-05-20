package com.example.gp2.ui.button;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import androidx.appcompat.app.AppCompatActivity;
import com.example.gp2.R;

public class DmDisplay extends AppCompatActivity {

    Button bt_dm;
    @Override
    public void onCreate(Bundle saveInstanceState)
    {
        super.onCreate(saveInstanceState);
        setContentView(R.layout.fragment_dm);

        bt_dm = findViewById(R.id.button_dm);
        bt_dm.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(DmDisplay.this, DmDisplay1.class);
                startActivity(intent);
            }
        });

    }
}
