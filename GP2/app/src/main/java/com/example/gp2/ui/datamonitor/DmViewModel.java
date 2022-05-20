package com.example.gp2.ui.datamonitor;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

public class DmViewModel extends ViewModel {

    private MutableLiveData<String> mText;

    public DmViewModel() {
        mText = new MutableLiveData<>();
        mText.setValue("点击按钮完成数据监测");
    }

    public LiveData<String> getText() {
        return mText;
    }
}