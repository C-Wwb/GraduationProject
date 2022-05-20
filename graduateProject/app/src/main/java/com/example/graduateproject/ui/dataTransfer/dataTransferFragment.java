package com.example.graduateproject.ui.dataTransfer;

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

public class dataTransferFragment extends Fragment {

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        com.example.graduateproject.ui.dataTransfer.dataTransferViewModel dataTransferViewModel = ViewModelProviders.of(this).get(com.example.graduateproject.ui.dataTransfer.dataTransferViewModel.class);
        View root = inflater.inflate(R.layout.fragment_datatransfer, container, false);
        final TextView textView = root.findViewById(R.id.text_dataTransfer);
        dataTransferViewModel.getText().observe(getViewLifecycleOwner(), new Observer<String>() {
            @Override
            public void onChanged(@Nullable String s) {
                textView.setText(s);
            }
        });
        return root;
    }
}