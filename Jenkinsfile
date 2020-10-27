

def cmake_build(compiler, flags, env4make, extradebugflags, prefixpath){
    def workspace_dir = pwd()
    def vcache = "/var/jenkins/.cache/miopen/vcache"
    def archive = (flags == '-DCMAKE_BUILD_TYPE=release')
    def config_targets = "check doc MIOpenDriver"
    def test_flags = "--disable-verification-cache"
    def debug_flags = "-g ${extradebugflags} -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined"
    def compilerpath = ""
    def configargs = ""
    if (prefixpath == "/usr/local")
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


pipeline {
    agent none 
    options {
        parallelsAlwaysFailFast()
    }
    environment{
        image = "miopen"
    }
    stages{
        // Run all static analysis tests
        stage("Static checks"){
            parallel{
                stage('Clang Tidy') {
                    agent{  label miopentest }
                    environment{
                        cmd = "rm -rf build; mkdir build; cd build; CXX='clang++-3.8' cmake -DBUILD_DEV=On ..; make -j\$(nproc) -k analyze;"
                    }
                    steps{
			env.HSA_ENABLE_SDMA=0
		        checkout scm
		        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
		        def dockerArgs = "--build-arg PREFIX=${prefixpath} -f hip-clang.docker "
		        def variant = env.STAGE_NAME
		        def retimage
		        gitStatusWrapper(credentialsId: '7126e5fe-eb51-4576-b52b-9aaf1de8f0fd', gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'MIOpen') {
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
                    	}
		}
            }
        }
    }    
}
}

