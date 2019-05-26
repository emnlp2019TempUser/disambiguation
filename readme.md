

# prepare data

```shell

    chmod +x preparedata.sh
    ./preparedata.sh

```


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



