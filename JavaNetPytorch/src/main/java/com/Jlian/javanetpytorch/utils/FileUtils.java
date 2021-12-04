package com.Jlian.javanetpytorch.utils;

import com.Jlian.javanetpytorch.constant.ViewModelConstants;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;

public class FileUtils {
    /**
     * 从输入流中获取数据
     * @param inStream 输入流
     * @return
     * @throws Exception
     */
    public static byte[] readInputStream(InputStream inStream) throws Exception{
        ByteArrayOutputStream outStream = new ByteArrayOutputStream();
        byte[] buffer = new byte[ViewModelConstants.MAX_FILE_SIZE];
        int len = 0;
        while( (len=inStream.read(buffer)) != -1 ){
            outStream.write(buffer, 0, len);
        }
        inStream.close();
        return outStream.toByteArray();
    }

}
