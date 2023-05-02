defmodule HNSWLib.BFIndex.Test do
  use ExUnit.Case
  doctest HNSWLib.BFIndex

  test "HNSWLib.BFIndex.new/3 with L2-space" do
    space = :l2
    dim = 8
    max_elements = 200
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim

    dim = 12
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim

    space = :cosine
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim

    space = :ip
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim
  end

  test "HNSWLib.BFIndex.new/3 with cosine-space" do
    space = :cosine
    dim = 8
    max_elements = 200
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim

    dim = 12
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim
  end

  test "HNSWLib.BFIndex.new/3 with inner-product space" do
    space = :ip
    dim = 8
    max_elements = 200
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim

    dim = 12
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim
  end
end
