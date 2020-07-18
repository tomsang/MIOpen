

def rocmnode(name) {
    def node_name = 'rocmtest'
    if(name == 'fiji') {
        node_name = 'rocmtest && fiji';
    } else if(name == 'vega') {
        node_name = 'rocmtest && vega';
    } else if(name == 'vega10') {
        node_name = 'rocmtest && vega10';
    } else if(name == 'vega20') {
        node_name = 'rocmtest && vega20';
    } else if(name == 'gfx908') {
        node_name = 'gfx908';
    } else {
        node_name = name
    }
    return node_name
}



def cmake_build(compiler, flags, env4make, prefixpath){
    def workspace_dir = pwd()
    def vcache = "/var/jenkins/.cache/miopen/vcache"
    def archive = (flags == '-DCMAKE_BUILD_TYPE=release')
    def config_targets = "check doc MIOpenDriver"
    def test_flags = "--disable-verification-cache"
    def debug_flags = "-g -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined"
    def compilerpath = ""
    def configargs = ""
    if (prefixpath == "")
        compilerpath = compiler;
    else
    {
        compilerpath = prefixpath + "/bin/" + compiler
        configargs = "-DCMAKE_PREFIX_PATH=${prefixpath}"
    }

    if (archive == true) {
        config_targets = "package"
    }
    def cmd = """
        echo \$HSA_ENABLE_SDMA
        ulimit -c unlimited
        rm -rf build
        mkdir build
        cd build
        CXX=${compilerpath} CXXFLAGS='-Werror' cmake ${configargs} -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS='${test_flags}' -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}' ${flags} .. 
        MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS=1 CTEST_PARALLEL_LEVEL=4 MIOPEN_VERIFY_CACHE_PATH=${vcache} MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 ${env4make} dumb-init make -j\$(nproc) ${config_targets}
    """
    echo cmd
    sh cmd
    // Only archive from master or develop
    if (archive == true && (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master")) {
        archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
    }
}

def buildJob(compiler, flags, env4make, image, prefixpath="/opt/rocm", cmd = ""){

        env.HSA_ENABLE_SDMA=0 
        checkout scm
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        def dockerArgs = "--build-arg PREFIX=${prefixpath} "
        if(prefixpath == "")
        {
            dockerArgs = ""
        }
        def retimage
        try {
            retimage = docker.build("${image}", dockerArgs + '.')
            withDockerContainer(image: image, args: dockerOpts) {
                timeout(time: 5, unit: 'MINUTES')
                {
                    sh 'PATH="/opt/rocm/opencl/bin/x86_64/:$PATH" clinfo'
                }
            }
        } catch(Exception ex) {
            retimage = docker.build("${image}", dockerArgs + "--no-cache .")
            withDockerContainer(image: image, args: dockerOpts) {
                timeout(time: 5, unit: 'MINUTES')
                {
                    sh 'PATH="/opt/rocm/opencl/bin/x86_64/:$PATH" clinfo'
                }
            }
        }

        withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
            timeout(time: 5, unit: 'HOURS')
            {
                if(cmd == ""){
                    cmake_build(compiler, flags, env4make, prefixpath)
                }else{
                    sh cmd
                }
            }
        }
        return retimage
}

def buildHipClangJob(compiler, flags, env4make, image, prefixpath="/opt/rocm", cmd = ""){

        env.HSA_ENABLE_SDMA=0 
        checkout scm
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        def dockerArgs = "--build-arg PREFIX=${prefixpath} -f hip-clang.docker "
        def retimage
        try {
            retimage = docker.build("${image}", dockerArgs + '.')
            withDockerContainer(image: image, args: dockerOpts) {
                timeout(time: 5, unit: 'MINUTES')
                {
                    sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo'
                }
            }
        } catch(Exception ex) {
            retimage = docker.build("${image}", dockerArgs + "--no-cache .")
            withDockerContainer(image: image, args: dockerOpts) {
                timeout(time: 5, unit: 'MINUTES')
                {
                    sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo'
                }
            }
        }

        withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
            timeout(time: 5, unit: 'HOURS')
            {
                if(cmd == ""){
                    cmake_build(compiler, flags, env4make, prefixpath)
                }else{
                    sh cmd
                }
            }
        }
        return retimage
}



pipeline {
    agent none 
    options {
        parallelsAlwaysFailFast()
    }
    environment{
        image = "miopen"
    }
    stages{
        // Misc tests
        stage("Aux tests"){
            parallel{
                stage('Hip clang debug COMGR') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_USE_COMGR=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS=--disable-verification-cache .. 
                            CTEST_PARALLEL_LEVEL=2 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """

                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "MIOPEN_LOG_LEVEL=5 MIOPEN_COMPILE_PARALLEL_LEVEL=1",  image+'-hip-clang', "/usr/local", cmd)
                    }
                }

                stage('Hip clang debug COMGR copy 1 ') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_USE_COMGR=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS=--disable-verification-cache .. 
                            CTEST_PARALLEL_LEVEL=2 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """

                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "MIOPEN_LOG_LEVEL=5 MIOPEN_COMPILE_PARALLEL_LEVEL=1",  image+'-hip-clang', "/usr/local", cmd)
                    }
                }

                stage('Hip clang debug COMGR copy 2') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_USE_COMGR=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS=--disable-verification-cache .. 
                            CTEST_PARALLEL_LEVEL=2 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """

                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "MIOPEN_LOG_LEVEL=5 MIOPEN_COMPILE_PARALLEL_LEVEL=1",  image+'-hip-clang', "/usr/local", cmd)
                    }
                }
                stage('Hip clang debug COMGR copy 3') {
                    agent{ label rocmnode("vega20") }
                    environment{
                        cmd = """
                            ulimit -c unlimited
                            rm -rf build
                            mkdir build
                            cd build
                            CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_USE_COMGR=On -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug -DMIOPEN_GPU_SYNC=On -DMIOPEN_TEST_FLAGS=--disable-verification-cache .. 
                            CTEST_PARALLEL_LEVEL=2 MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM=0 MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0 make -j\$(nproc) check
                        """

                    }
                    steps{
                        buildHipClangJob('/opt/rocm/llvm/bin/clang++', '', "MIOPEN_LOG_LEVEL=5 MIOPEN_COMPILE_PARALLEL_LEVEL=1",  image+'-hip-clang', "/usr/local", cmd)
                    }
                }            }
        }
    }    
}

