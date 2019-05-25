

# prepare data

Please download data [here](https://static.aminer.cn/misc/na-data-kdd18.zip) (or via [OneDrive](https://1drv.ms/u/s!AjyjU4F_oXtllmRV9aFPN1bpkEBY)). Unzip the file and put the _data_ directory into project's data dictionary directory.


# preprocess

``` python

    export PYTHONPATH="/Users/csx/GitProject/Research/disambiguation:$PYTHONPATH"


    python preprocess/genAuthorIds.py
    python preprocess/prepareData.py

```

# Run

```

    python DualGAE.py

```

# Result

the result data is stored in ./data/result/result.txt

![image.png](https://upload-images.jianshu.io/upload_images/5786775-26cd0942181d76f4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



