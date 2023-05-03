# hnswlib_elixir

Elixir binding for the [hnswlib](https://github.com/nmslib/hnswlib) library.

## Usage
### Create an Index
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
{:ok,
 %HNSWLib.Index{
   space: :l2,
   dim: 2,
   reference: #Reference<0.2548668725.3381002243.154990>
 }}
```

### Add vectors to the Index
```elixir
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
iex> HNSWLib.Index.get_current_count(index)
{:ok, 0}
iex> HNSWLib.Index.add_items(index, data)
:ok
iex> HNSWLib.Index.get_current_count(index)
{:ok, 5}
```

### Query nearest vector(s) in the Index
```elixir
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

iex> {:ok, labels, dists} = HNSWLib.Index.knn_query(index, query, k: 3)
{:ok,
 #Nx.Tensor<
   u64[1][3]
   [
     [2, 0, 1]
   ]
 >,
 #Nx.Tensor<
   f32[1][3]
   [
     [5.0, 3281.0, 3445.0]
   ]
 >}
```

### Save an Index to file
```elixir
iex> HNSWLib.Index.save_index(index, "my_index.bin")
:ok
```

### Load an Index from file
```elixir
iex> {:ok, saved_index} = HNSWLib.Index.load_index(space, dim, "my_index.bin")
{:ok,
 %HNSWLib.Index{
   space: :l2,
   dim: 2,
   reference: #Reference<0.2105700569.2629697564.236704>
 }}
iex> HNSWLib.Index.get_current_count(saved_index)
{:ok, 5}
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

