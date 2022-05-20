package com.example.graduateproject.ui.dataDetection;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

public class dataDetectionViewModel extends ViewModel {

    private MutableLiveData<String> mText;

    public dataDetectionViewModel() {
        mText = new MutableLiveData<>();
        mText.setValue("This is data detection fragment");
    }

    public LiveData<String> getText() {
        return mText;
    }
}