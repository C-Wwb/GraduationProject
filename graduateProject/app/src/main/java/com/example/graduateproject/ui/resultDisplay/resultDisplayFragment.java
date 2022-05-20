package com.example.graduateproject.ui.resultDisplay;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProviders;

import com.example.graduateproject.R;

public class resultDisplayFragment extends Fragment {

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        com.example.graduateproject.ui.resultDisplay.resultDisplayViewModel resultDisplayViewModel = ViewModelProviders.of(this).get(com.example.graduateproject.ui.resultDisplay.resultDisplayViewModel.class);
        View root = inflater.inflate(R.layout.fragment_resultdisplay, container, false);
        final TextView textView = root.findViewById(R.id.text_resultDisplay);
        resultDisplayViewModel.getText().observe(getViewLifecycleOwner(), new Observer<String>() {
            @Override
            public void onChanged(@Nullable String s) {
                textView.setText(s);
            }
        });
        return root;
    }
}