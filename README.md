# hnswlib_elixir

Elixir binding for the [hnswlib](https://github.com/nmslib/hnswlib) library.

## Usage
```elixir
# working in L2-space
# other possible values are
#  `:ip` (inner product)
#  `:cosine` (cosine similarity)
iex> space = :l2
:l2
# each vector is a 2D-vec
iex> dim = 2
2
# limit the maximum elements to 200
iex> max_elements = 200
200
# create Index
iex> {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)
{:ok, %HNSWLib.Index{space: :l2, dim: 2, pid: #PID<0.242.0>}}

# add some vectors
iex> data =
  Nx.tensor(
    [
      [42, 42],
      [43, 43],
      [0, 0],
      [200, 200],
      [200, 220]
    ],
    type: :f32
  )
#Nx.Tensor<
  f32[5][2]
  [
    [42.0, 42.0],
    [43.0, 43.0],
    [0.0, 0.0],
    [200.0, 200.0],
    [200.0, 220.0]
  ]
>
iex> HNSWLib.Index.add_items(index, data)
:ok

# query
iex> query = Nx.tensor([1, 2], type: :f32)
#Nx.Tensor<
  f32[2]
  [1.0, 2.0]
>
iex> {:ok, labels, dists} = HNSWLib.Index.knn_query(index, query)
{:ok,
 #Nx.Tensor<
   u64[1][1]
   [
     [2]
   ]
 >,
 #Nx.Tensor<
   f32[1][1]
   [
     [5.0]
   ]
 >}
```

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `hnswlib_elixir` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:hnswlib_elixir, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/hnswlib_elixir>.

