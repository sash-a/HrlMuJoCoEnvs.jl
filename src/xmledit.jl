using LightXML

file = joinpath(@__DIR__, "..", "assets", "ant.xml")
xdoc = parse_file(file)

xroot = root(xdoc)
worldbody = find_element(xroot, "worldbody")
name(worldbody)
# value(worldbody)
c = new_child(worldbody, "TestChild")
w = 123
set_attributes(c; testgeom="somegeom", dunno="hellow $w")

save_file(xdoc, "test.xml")

free(xdoc)