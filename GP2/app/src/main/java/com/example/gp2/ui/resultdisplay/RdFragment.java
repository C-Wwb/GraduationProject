package com.example.gp2.ui.resultdisplay;

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

public class RdFragment extends Fragment {

    private RdViewModel rdViewModel;

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        rdViewModel =
                ViewModelProviders.of(this).get(RdViewModel.class);
        View root = inflater.inflate(R.layout.fragment_rd, container, false);
        final TextView textView = root.findViewById(R.id.text_rd);
        rdViewModel.getText().observe(getViewLifecycleOwner(), new Observer<String>() {
            @Override
            public void onChanged(@Nullable String s) {
                textView.setText(s);
            }
        });
        return root;
    }
}