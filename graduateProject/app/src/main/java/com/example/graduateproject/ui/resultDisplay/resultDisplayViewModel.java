package com.example.graduateproject.ui.resultDisplay;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

public class resultDisplayViewModel extends ViewModel {

    private MutableLiveData<String> mText;

    public resultDisplayViewModel() {
        mText = new MutableLiveData<>();
        mText.setValue("This is notifications fragment");
    }

    public LiveData<String> getText() {
        return mText;
    }
}