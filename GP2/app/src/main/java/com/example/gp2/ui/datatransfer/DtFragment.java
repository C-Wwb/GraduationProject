package com.example.gp2.ui.datatransfer;

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

import com.example.gp2.R;

public class DtFragment extends Fragment {

    private DtViewModel dtViewModel;

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        dtViewModel =
                ViewModelProviders.of(this).get(DtViewModel.class);
        View root = inflater.inflate(R.layout.fragment_dt, container, false);
        final TextView textView = root.findViewById(R.id.text_dt);
        dtViewModel.getText().observe(getViewLifecycleOwner(), new Observer<String>() {
            @Override
            public void onChanged(@Nullable String s) {
                textView.setText(s);
            }
        });
        return root;
    }
}