package com.Jlian.javanetpytorch.service;

public interface ImageQueueService {
    String detectQueue(String uuid, String base64EncoderImage) throws InterruptedException;
}
