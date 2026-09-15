ret = aclnnMatmulGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    ret = aclSetAclOpExecutorRepeatable(executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclSetAclOpExecutorRepeatable failed. ERROR: %d\n", ret); return ret);





constexpr int warmup = 10;
    constexpr int repeat = 100;

    for (int i = 0; i < warmup; ++i) {
      ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclrtEvent startEvent = nullptr;
    aclrtEvent endEvent = nullptr;

    ret = aclrtCreateEvent(&startEvent);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtCreateEvent(&endEvent);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtRecordEvent(startEvent, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    for (int i = 0; i < repeat; ++i) {
      ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclrtRecordEvent(endEvent, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtSynchronizeEvent(endEvent);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    float totalMs = 0.0F;
    ret = aclrtEventElapsedTime(&totalMs, startEvent, endEvent);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    LOG_PRINT(
        "MATMUL_LATENCY average_ms=%.9f repeat=%d\n",
        totalMs / repeat,
        repeat);

    aclrtDestroyEvent(endEvent);
    aclrtDestroyEvent(startEvent);
