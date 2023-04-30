defmodule HNSWLib.Index do
  @moduledoc """
  Documentation for `HNSWLib.Index`.
  """

  defstruct [:space, :dim, :reference]
  alias __MODULE__, as: T

  @doc """
  Construct a new Index

  ##### Positional Parameters

  - *space*: `:cosine` | `:ip` | `:l2`.

    An atom that indicates the vector space. Valid values are

      - `:cosine`, cosine space
      - `:ip`, inner product space
      - `:l2`, L2 space

  - *dim*: `non_neg_integer()`.

    Number of dimensions for each vector.
  """
  @spec new(:cosine | :ip | :l2, non_neg_integer()) :: {:ok, %T{}} | {:error, String.t()}
  def new(space, dim)
      when (space == :l2 or space == :ip or space == :cosine) and is_integer(dim) and dim >= 0 do
    case HNSWLib.Nif.new(space, dim) do
      {:ok, ref} when is_reference(ref) ->
        {:ok,
         %T{
           space: space,
           dim: dim,
           reference: ref
         }}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Init the given Index

  An Index should only be inited once.

  ##### Positional Parameters

  - *self*: `%HNSWLib.Index{}`.

    Index variable.

  - *max_elements*: `non_neg_integer()`.

    Number of maximum elements.

  ##### Keyword Paramters

  - *m*: `non_neg_integer()`.
  - *ef_construction*: `non_neg_integer()`.
  - *random_seed*: `non_neg_integer()`.
  - *allow_replace_deleted*: `boolean()`.
  """
  @spec init_index(%T{:reference => reference()}, non_neg_integer(), [
          {:m, non_neg_integer()},
          {:ef_construction, non_neg_integer()},
          {:random_seed, non_neg_integer()},
          {:allow_replace_deleted, boolean()}
        ]) :: :ok | {:error, String.t()}
  def init_index(%T{reference: self}, max_elements, opts \\ [])
      when is_reference(self) and is_integer(max_elements) and max_elements >= 0 do
    m = opts[:m] || 16
    ef_construction = opts[:ef_construction] || 200
    random_seed = opts[:random_seed] || 100
    allow_replace_deleted = opts[:allow_replace_deleted] || false

    HNSWLib.Nif.init_index(
      self,
      max_elements,
      m,
      ef_construction,
      random_seed,
      allow_replace_deleted
    )
  end
end
