cmake_minimum_required(VERSION 3.20)
project(mind_sim LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(MindSimBuildConfig)
include(MindSimDependencies)
include(MindSimModCodegen)
include(MindSimMechanisms)
include(MindSimSources)
include(MindSimPythonModule)
include(MindSimTests)
