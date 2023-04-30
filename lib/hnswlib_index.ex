defmodule HNSWLib.Index do
  @moduledoc """
  Documentation for `HNSWLib.Index`.
  """

  defstruct [:space, :dim, :reference]
  alias __MODULE__, as: T

  @doc """
  Construct a new Index

  ##### Positional Parameters

  - *space*, `:cosine` | `:ip` | `:l2`.

    An atom that indicates the vector space. Valid values are

      - `:cosine`, cosine space
      - `:ip`, inner product space
      - `:l2`, L2 space

  - *dim*, `non_neg_integer()`.

    Number of dimensions for each vector.
  """
  @spec new(:cosine | :ip | :l2, non_neg_integer()) :: {:ok, %T{}} | {:error, String.t()}
  def new(space, dim)
  when (space == :l2 or space == :ip or space == :cosine) and is_integer(dim) and dim >= 0 do
    case HNSWLib.Nif.new(space, dim) do
      {:ok, ref} when is_reference(ref) ->
        {:ok, %T{
          space: space,
          dim: dim,
          reference: ref
        }}

      {:error, reason} ->
        {:error, reason}
    end
  end
end
