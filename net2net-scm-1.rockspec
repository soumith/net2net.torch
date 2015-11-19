package = "net2net"
version = "scm-1"

source = {
url = "git://github.com/soumith/net2net.torch",
tag = "master"
}

description = {
summary = "Implementation of net2net transforms in Torch",
detailed = [[
As described in the paper: Net2Net: Accelerating Learning via Knowledge Transfer http://arxiv.org/abs/1511.05641
]],
homepage = "https://github.com/soumith/net2net.torch",
license = "BSD"
}

dependencies = {
"torch >= 7.0",
"nn",
}

build = {
type = "command",
build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
install_command = "cd build && $(MAKE) install"
}