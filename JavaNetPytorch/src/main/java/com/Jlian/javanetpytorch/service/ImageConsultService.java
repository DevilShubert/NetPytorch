package com.Jlian.javanetpytorch.service;

import org.springframework.web.multipart.MultipartFile;

import java.io.FileNotFoundException;
import java.io.IOException;

public interface ImageConsultService {
    String consultByFile(MultipartFile imageFile) throws IOException, InterruptedException;
}
