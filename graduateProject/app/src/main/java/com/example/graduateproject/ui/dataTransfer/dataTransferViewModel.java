package com.example.graduateproject.ui.dataTransfer;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

public class dataTransferViewModel extends ViewModel {

    private MutableLiveData<String> mText;

    public dataTransferViewModel() {
        mText = new MutableLiveData<>();
        mText.setValue("This is dashboard fragment");
    }

    public LiveData<String> getText() {
        return mText;
    }
}